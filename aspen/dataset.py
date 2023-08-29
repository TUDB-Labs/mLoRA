from aspen.modelargs import MultiLoraBatchData, LoraBatchDataConfig
from aspen.tokenizer import Tokenizer

import math
import json
import random
from typing import Dict, List, Tuple


class DataSet():
    config_ = None
    # Dict[lora_name, <prompt, token>]
    lora_token_data_: Dict[str, List[Tuple[str, List[int]]]] = None

    lora_num_epochs_: Dict[str, int] = {}
    lora_cnt_epochs_: Dict[str, int] = {}
    lora_start_idx_: Dict[str, int] = {}
    lora_batch_size_: Dict[str, int] = {}

    def __get_lora_text_data(self) -> Dict[str, List[str]]:
        lora_text_data = {}
        for lora_config in self.config_["lora"]:
            lora_name = lora_config["name"]
            data_path = lora_config["data"]
            lora_text_data[lora_name] = []
            self.lora_cnt_epochs_[lora_name] = 0
            self.lora_start_idx_[lora_name] = 0
            self.lora_batch_size_[lora_name] = lora_config["batch_size"]
            self.lora_num_epochs_[lora_name] = lora_config["num_epochs"]

            with open(data_path, 'r', encoding='utf8') as fp:
                for raw_data in json.load(fp):
                    raw_data_input = raw_data["input"]
                    raw_data_output = raw_data["output"]
                    raw_data_instruction = raw_data["instruction"]
                    text_data = ""
                    if raw_data_input is None or len(raw_data_input) <= 1:
                        text_data = lora_config["prompt_no_input"].replace(
                            "{output}", raw_data_output).replace("{instruction}", raw_data_instruction)
                    else:
                        text_data = lora_config["prompt_input"].replace(
                            "{output}", raw_data_output).replace(
                            "{instruction}", raw_data_instruction).replace(
                            "{input}", raw_data_input)
                    lora_text_data[lora_name].append(text_data)

        return lora_text_data

    def __init__(self, config: Dict[str, str], tokenizer: Tokenizer):
        self.config_ = config
        self.tokenizer_: Tokenizer = tokenizer

        print("to load text data from file.")
        lora_text_data = self.__get_lora_text_data()
        print("load text data from file done.")

        # Dict[lora_name, <prompt, token>]
        self.lora_token_data_: Dict[str, List[Tuple[str, List[int]]]] = {}

        print("to encode text data to tokens")
        for lora_name in lora_text_data:
            self.lora_token_data_[lora_name] = []

            for idx, text in enumerate(lora_text_data[lora_name]):
                tokens = tokenizer.encode(text, bos=True, eos=True)
                if len(tokens) > config["cutoff_len"]:
                    tokens = tokens[:config["cutoff_len"]]
                self.lora_token_data_[lora_name].append((text, tokens))
                if idx % 10000 == 0:
                    print(
                        f"encode text data: {idx}/{len(lora_text_data[lora_name])}")
            # group by length
            if self.config_["group_by_length"]:
                self.lora_token_data_[lora_name].sort(
                    key=lambda x: len(x[1]), reverse=True)
            else:
                random.shuffle(self.lora_token_data_[lora_name])
        print("encode text data to tokens done.")

    def check_done(self) -> bool:
        for lora_name in self.lora_token_data_:
            if self.lora_cnt_epochs_[lora_name] < self.lora_num_epochs_[lora_name]:
                return False
        return True

    def get_batch_data(self) -> MultiLoraBatchData:
        prompts_list: List[str] = []
        batch_tokens_list: List[List[int]] = []

        prompts_batch_config_list: List[LoraBatchDataConfig] = []

        tokens_without_pad_len_list: List[int] = []

        max_token_len = 0

        batch_start_idx = 0

        for lora_name in self.lora_token_data_:
            if self.lora_cnt_epochs_[lora_name] >= self.lora_num_epochs_[lora_name]:
                continue
            start_idx = self.lora_start_idx_[lora_name]
            end_idx = start_idx + self.lora_batch_size_[lora_name]
            prompt_and_tokens_list = self.lora_token_data_[
                lora_name][start_idx:end_idx]

            for pt in prompt_and_tokens_list:
                prompt, token = pt
                prompts_list.append(prompt)
                batch_tokens_list.append(token.copy())

                max_token_len = max(max_token_len, len(token))
                tokens_without_pad_len_list.append(len(token))

            lora_config = LoraBatchDataConfig(adapter_name_=lora_name, batch_start_idx_=batch_start_idx,
                                              batch_end_idx_=batch_start_idx + len(prompt_and_tokens_list))
            batch_start_idx += len(prompt_and_tokens_list)
            prompts_batch_config_list.append(lora_config)

            self.lora_start_idx_[lora_name] += self.lora_batch_size_[lora_name]
            if self.lora_start_idx_[lora_name] >= len(self.lora_token_data_[lora_name]):
                self.lora_start_idx_[lora_name] = 0
                self.lora_cnt_epochs_[lora_name] += 1

            print(f"{lora_name} train data:")
            print(
                f"    epoch: {self.lora_cnt_epochs_[lora_name] + 1} / {self.lora_num_epochs_[lora_name]}")
            print(
                f"    step : {self.lora_start_idx_[lora_name]} / {len(self.lora_token_data_[lora_name])}")

        # align batch data
        max_token_len = math.ceil(max_token_len / 8) * 8

        for tokens in batch_tokens_list:
            while len(tokens) < max_token_len:
                if self.config_["expand_right"]:
                    tokens.append(self.tokenizer_.pad_id_)
                else:
                    tokens.insert(0, self.tokenizer_.pad_id_)

        print(
            f"batch data size: {max_token_len} * {len(batch_tokens_list)}")

        return MultiLoraBatchData(prompts_=prompts_list,
                                  lora_batch_data_config_=prompts_batch_config_list,
                                  batch_seq_len_=max_token_len,
                                  expand_right_=self.config_["expand_right"],
                                  batch_tokens_=batch_tokens_list,
                                  tokens_len_without_pad_=tokens_without_pad_len_list)
