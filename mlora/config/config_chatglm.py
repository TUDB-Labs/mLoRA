from transformers import PretrainedConfig


class ChatGLMConfig:
    model_type = "chatglm"
    _name_or_path: str
    add_bias_linear = False,
    add_qkv_bias = False,
    apply_query_key_layer_scaling = True,
    apply_residual_connection_post_layernorm = False,
    attention_dropout = 0.0,
    attention_softmax_in_fp32 = True,
    bias_dropout_fusion = True,
    classifier_dropout = None,
    eos_token_id = None,
    ffn_hidden_size = 13696,
    fp32_residual_connection = False,
    hidden_dropout = 0.0,
    hidden_size = 4096,
    kv_channels = 128,
    layernorm_epsilon = 1.5625e-07,
    model_type = "chatglm",
    multi_query_attention = True,
    multi_query_group_num = 2,
    norm_eps_ = 1.5625e-07,
    num_attention_heads = 32,
    num_hidden_layers = 40,
    num_layers = 40,
    original_rope = False,
    output_features = 151552,
    pad_token_id = 151329,
    padded_vocab_size = 151552,
    post_layer_norm = True,
    rmsnorm = True,
    rope_ratio = 500,
    seq_length = 131072,
    tie_word_embeddings = False,
    torch_dtype = None,
    vocab_size = 151552,
    _attn_implementation = "default"

    def __init__(self, config: PretrainedConfig, **kwargs):
        self._name_or_path = config._name_or_path
        self.num_layers = config.num_layers
        self.vocab_size = config.padded_vocab_size
        self.padded_vocab_size = config.padded_vocab_size
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.kv_channels = config.kv_channels
        self.num_attention_heads = config.num_attention_heads
        self.seq_length = config.seq_length
        self.hidden_dropout = config.hidden_dropout
        self.classifier_dropout = config.classifier_dropout
        self.attention_dropout = config.attention_dropout
        self.layernorm_epsilon = config.layernorm_epsilon
        self.rmsnorm = config.rmsnorm
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)
        self.post_layer_norm = config.post_layer_norm
        self.add_bias_linear = config.add_bias_linear
        self.add_qkv_bias = config.add_qkv_bias
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.multi_query_attention = config.multi_query_attention
        self.multi_query_group_num = config.multi_query_group_num
        self.rope_ratio = config.rope_ratio
        self.apply_query_key_layer_scaling = (
            config.apply_query_key_layer_scaling)
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.fp32_residual_connection = config.fp32_residual_connection
        self.norm_eps_ = config.layernorm_epsilon
        self.dim_ = config.hidden_size
        self.vocab_size_ = config.vocab_size
        self.eos_token_id = config.eos_token_id
        self.model_type = config.model_type
        self.num_hidden_layers = config.num_hidden_layers
        self.original_rope = config.original_rope
        self.pad_token_id = config.pad_token_id
        self.tie_word_embeddings = config.tie_word_embeddings
        self.torch_dtype = config.torch_dtype
        self._attn_implementation = config._attn_implementation
        super().__init__(**kwargs)
