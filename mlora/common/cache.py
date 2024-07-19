import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .model import Cache
from .modelargs import LLMModelConfig


class DynamicCache(Cache):
    def __init__(self, **kwargs) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search.
        """

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(
        self, full_batch_size: int, split_size: int
    ) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [
                tensor[i : i + split_size] for tensor in self.key_cache
            ]
            current_split.value_cache = [
                tensor[i : i + split_size] for tensor in self.value_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat(
                [current.key_cache[idx] for current in splits], dim=0
            )
            layer_values = torch.cat(
                [current.value_cache[idx] for current in splits], dim=0
            )
            cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class StaticCache(Cache):
    def __init__(
        self,
        config: LLMModelConfig,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype=None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = (
            config.max_seq_len_ if max_cache_len is None else max_cache_len
        )
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = config.head_dim_

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = config.n_kv_heads_

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (
            max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        for _ in range(config.n_layers_):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            new_layer_value_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            k_out[:, :, cache_position] = key_states
            v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


class SlidingWindowCache(StaticCache):
    def __init__(
        self,
        config: LLMModelConfig,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype=None,
    ) -> None:
        if not hasattr(config, "sliding_window_") or config.sliding_window_ is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        max_cache_len = min(config.sliding_window_, max_cache_len)
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # assume this only happens in prefill phase when prompt length > sliding_window_size (= max_cache_len)
        if cache_position.shape[0] > self.max_cache_len:
            k_out = key_states[:, :, -self.max_cache_len :, :]
            v_out = value_states[:, :, -self.max_cache_len :, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(
            self.max_cache_len, dtype=torch.long, device=value_states.device
        ).cumsum(0)
        cache_position = cache_position.clamp(0, self.max_cache_len - 1)
        to_shift = cache_position >= self.max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % self.max_cache_len

        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out

        return k_out, v_out

    def get_max_length(self) -> Optional[int]:
        # in theory there is no limit because the sliding window size is fixed no matter how long the sentence is
        return None

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


class HybridCache(Cache):
    def __init__(
        self,
        config: LLMModelConfig,
        max_batch_size,
        max_cache_len,
        device="cpu",
        dtype=None,
    ) -> None:
        if not hasattr(config, "sliding_window_") or config.sliding_window_ is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = config.head_dim_

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = config.n_kv_heads_
        self.is_sliding = torch.tensor(
            [not bool(i % 2) for i in range(config.n_layers_)],
            dtype=torch.bool,
            device=device,
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (
            max_batch_size,
            self.num_key_value_heads,
            max_cache_len,
            self.head_dim,
        )
        sliding_cache_shape = (
            max_batch_size,
            self.num_key_value_heads,
            min(config.sliding_window_, max_cache_len),
            self.head_dim,
        )
        for i in range(config.n_layers_):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            cache_shape = (
                global_cache_shape if not self.is_sliding[i] else sliding_cache_shape
            )
            new_layer_key_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            new_layer_value_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def _sliding_update(
        self,
        cache_position,
        layer_idx,
        key_states,
        value_states,
        k_out,
        v_out,
        max_cache_len,
    ):
        if cache_position.shape[0] > max_cache_len:
            k_out = key_states[:, :, -max_cache_len:, :]
            v_out = value_states[:, :, -max_cache_len:, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(
            max_cache_len, dtype=torch.long, device=value_states.device
        ).cumsum(0)
        cache_position = cache_position.clamp(0, max_cache_len - 1)
        to_shift = cache_position >= max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len
        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out
        return k_out, v_out

    def _static_update(
        self,
        cache_position,
        layer_idx,
        key_states,
        value_states,
        k_out,
        v_out,
        max_cache_len,
    ):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        sliding_window = cache_kwargs.get("sliding_window")
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(
            device=key_states.device
        )
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(
            device=value_states.device
        )
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        if sliding_window:
            update_fn = self._sliding_update
        else:
            update_fn = self._static_update

        return update_fn(
            cache_position,
            layer_idx,
            key_states,
            value_states,
            k_out,
            v_out,
            k_out.shape[2],
        )

    def get_max_length(self) -> Optional[int]:
        # in theory there is no limit because the sliding window size is fixed
        # no matter how long the sentence is
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        return None

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


cache_dict = {
    "dynamic": DynamicCache,
    "static": StaticCache,
    "sliding_window": SlidingWindowCache,
    "hybrid": HybridCache,
}


def cache_factory(
    cache_implementation: str,
    config: LLMModelConfig,
    max_batch_size: int,
    max_cache_len: int,
):
    assert (
        cache_implementation in cache_dict
    ), f"Unknown cache type. {cache_implementation}"
    logging.info(f"Use {cache_implementation} as cache implementation.")
    if cache_implementation == "sliding_window":
        assert hasattr(config, "sliding_window_")
        max_cache_len = min(config.sliding_window_, max_cache_len)
    return cache_dict[cache_implementation](
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        device=config.device_,
        dtype=config.dtype_,
    )
