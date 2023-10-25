from aspen.modelargs import MultiLoraBatchData, LoraBatchDataConfig
from aspen.tokenizer import Tokenizer

import sys
import math
import json
import random
import datasets
from typing import Dict, List, Tuple


def load_dataset(data_path: str):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        return datasets.load_dataset("text", data_files={"train": data_path})
    else:
        return datasets.load_dataset(data_path)


class DataSet():
    config_ = None
    tokenizer_: Tokenizer = None

    # Dict[lora_name, <prompt, token>]
    lora_token_data_: Dict[str, List[Tuple[str, List[int]]]] = None
    # Dict[lora_name, <prompt, token>]
    lora_test_token_data_: Dict[str, List[Tuple[str, List[int]]]] = None

    lora_num_epochs_: Dict[str, int] = {}
    # count the epoch, if all train data done, the epoch + 1
    lora_cnt_epochs_: Dict[str, int] = {}
    # next train data start position, the position = old position + batchsize
    lora_start_idx_: Dict[str, int] = {}
    lora_batch_size_: Dict[str, int] = {}

    lora_test_data_batch_size_: Dict[str, int] = {}
    lora_test_data_start_index_: Dict[str, int] = {}

    # read text data and template data
    def __parse_data_with_template(self,
                                   data: List,
                                   template_parameter_list: List[str],
                                   template_prompt_no_input: str,
                                   template_prompt: str) -> List[str]:
        ret_text_data = []
        for raw_data in data:
            raw_data_input = {}

            no_input_flag = False
            for para in template_parameter_list:
                if para not in raw_data:
                    no_input_flag = True
                    continue
                raw_data_input[para] = raw_data[para]

            text_data: str = ""
            if no_input_flag:
                text_data = template_prompt_no_input
            else:
                text_data = template_prompt

            for para in template_parameter_list:
                if para not in raw_data_input:
                    continue
                text_data = text_data.replace(
                    "{" + para + "}", raw_data[para])
            ret_text_data.append(text_data)
        return ret_text_data

    # get different train data and test data
    # the output type is Dict[lora_name, prompt]
    def __get_lora_text_data(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        lora_train_text_data = {}
        lora_test_text_data = {}

        for lora_config in self.config_["lora"]:
            lora_name = lora_config["name"]
            lora_template = lora_config["prompt"]
            data_path = lora_config["data"]
            test_data_path = lora_config.get("test_data", None)
            val_set_size = lora_config.get("val_set_size", -1)

            with open(lora_template, 'r', encoding='utf8') as fp:
                template_config = json.load(fp)

            template_parameter_list = template_config["parameter"]
            template_prompt = template_config["prompt"]
            template_prompt_no_input = template_config["prompt_no_input"]

            self.lora_cnt_epochs_[lora_name] = 0
            self.lora_start_idx_[lora_name] = 0
            self.lora_batch_size_[lora_name] = lora_config["micro_batch_size"]
            self.lora_num_epochs_[lora_name] = lora_config["num_epochs"]

            self.lora_test_data_batch_size_[
                lora_name] = lora_config["test_batch_size"]
            self.lora_test_data_start_index_[lora_name] = 0

            data = load_dataset(data_path)
            if test_data_path is None:
                train_val = data["train"].train_test_split(test_size=val_set_size)
                train_text_data = self.__parse_data_with_template(train_val["train"].
                                                                  template_parameter_list,
                                                                  template_prompt_no_input,
                                                                  template_prompt)
                test_text_data = self.__parse_data_with_template(train_val["test"],
                                                                 template_parameter_list,
                                                                 template_prompt_no_input,
                                                                 template_prompt)
            else:
                train_data = load_dataset(test_data_path)
                train_text_data = self.__parse_data_with_template(data["train"],
                                                                  template_parameter_list,
                                                                  template_prompt_no_input,
                                                                  template_prompt)
                test_text_data = self.__parse_data_with_template(train_data["train"],
                                                                 template_parameter_list,
                                                                 template_prompt_no_input,
                                                                 template_prompt)

            if lora_name not in lora_train_text_data:
                lora_train_text_data[lora_name] = []
                lora_test_text_data[lora_name] = []
            lora_train_text_data[lora_name].extend(train_text_data)
            lora_test_text_data[lora_name].extend(test_text_data)

        return lora_train_text_data, lora_test_text_data

    # Output is Dict[lora_name, <prompt, token>]
    def __encode_prompt(self,
                        lora_text_data: Dict[str, List[str]],
                        config: Dict[str, str],
                        is_train_data: bool = True) -> Dict[str, List[Tuple[str, List[int]]]]:
        target: Dict[str, List[Tuple[str, List[int]]]] = {}
        for lora_name in lora_text_data:
            target[lora_name] = []
            for idx, text in enumerate(lora_text_data[lora_name]):
                tokens = self.tokenizer_.encode(text, bos=True, eos=True)
                if is_train_data:
                    if len(tokens) > config["cutoff_len"]:
                        tokens = tokens[:config["cutoff_len"]]

                target[lora_name].append((text, tokens))
                if idx % 10000 == 0:
                    print(
                        f"encode text data {lora_name}: {idx}/{len(lora_text_data[lora_name])}")

            # only train data need group by length or shuffle
            if is_train_data:
                if self.config_["group_by_length"]:
                    target[lora_name].sort(
                        key=lambda x: len(x[1]), reverse=True)
                else:
                    random.shuffle(target[lora_name])

        return target

    def __init__(self, config: Dict[str, str], tokenizer: Tokenizer):
        self.config_ = config
        self.tokenizer_: Tokenizer = tokenizer

        print("to load text data from file.")
        lora_train_text_data, lora_test_text_data = self.__get_lora_text_data()
        print("load text data from file done.")

        # Dict[lora_name, <prompt, token>]
        print("to encode train text data to tokens")
        self.lora_token_data_ = self.__encode_prompt(
            lora_train_text_data, config, is_train_data=True)
        print("encode train text data to tokens done.")

        print("to encode test text data to tokens")
        self.lora_test_token_data_ = self.__encode_prompt(
            lora_test_text_data, config, is_train_data=True)
        print("encode test text data to tokens done.")

    def check_done(self) -> bool:
        for lora_name in self.lora_token_data_:
            if self.lora_cnt_epochs_[lora_name] < self.lora_num_epochs_[lora_name]:
                return False
        return True

    def check_test_done(self) -> bool:
        for lora_name in self.lora_test_token_data_:
            if self.lora_test_data_start_index_[lora_name] < len(self.lora_test_token_data_[lora_name]):
                return False
        return True

    def get_test_barch_data(self) -> MultiLoraBatchData:
        prompts_list: List[str] = []
        batch_tokens_list: List[List[int]] = []

        prompts_batch_config_list: List[LoraBatchDataConfig] = []

        min_token_len: int = sys.maxsize
        max_token_len: int = -1

        batch_start_idx = 0

        for lora_name in self.lora_test_token_data_:
            if self.lora_test_data_start_index_[lora_name] >= len(self.lora_test_token_data_[lora_name]):
                continue
            start_idx = self.lora_test_data_start_index_[lora_name]
            end_idx = start_idx + self.lora_test_data_batch_size_[lora_name]
            # get the prompts and tokens list to test
            prompt_and_tokens_list = self.lora_test_token_data_[
                lora_name][start_idx:end_idx]
            for pt in prompt_and_tokens_list:
                prompt, token = pt
                prompts_list.append(prompt)
                batch_tokens_list.append(token.copy())

                max_token_len = max(max_token_len, len(token))
                min_token_len = min(min_token_len, len(token))

            lora_config = LoraBatchDataConfig(adapter_name_=lora_name, batch_start_idx_=batch_start_idx,
                                              batch_end_idx_=batch_start_idx + len(prompt_and_tokens_list))
            batch_start_idx += len(prompt_and_tokens_list)
            prompts_batch_config_list.append(lora_config)

            print(f"{lora_name} test data: ")
            print(
                f"    count: {self.lora_test_data_start_index_[lora_name]} / {len(self.lora_test_token_data_[lora_name])}")

            self.lora_test_data_start_index_[
                lora_name] += self.lora_test_data_batch_size_[lora_name]

        # append pad to align
        for tokens in batch_tokens_list:
            while len(tokens) < self.config_["cutoff_len"]:
                tokens.append(self.tokenizer_.eos_id_)

        return MultiLoraBatchData(prompts_=prompts_list,
                                  lora_batch_data_config_=prompts_batch_config_list,
                                  batch_seq_len_=self.config_[
                                      "cutoff_len"],
                                  expand_right_=True,
                                  batch_tokens_=batch_tokens_list,
                                  tokens_len_without_pad_=[],
                                  inference_model_=True,
                                  cache_key_=[],
                                  cache_value_=[],
                                  min_token_size_=min_token_len,
                                  max_token_size_=max_token_len,
                                  max_cutoff_len_=self.config_[
                                      "cutoff_len"]
                                  )

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

            print(f"{lora_name} train data:")
            print(
                f"    epoch: {self.lora_cnt_epochs_[lora_name] + 1} / {self.lora_num_epochs_[lora_name]}")
            print(
                f"    step : {self.lora_start_idx_[lora_name]} / {len(self.lora_token_data_[lora_name])}")

            self.lora_start_idx_[lora_name] += self.lora_batch_size_[lora_name]
            if self.lora_start_idx_[lora_name] >= len(self.lora_token_data_[lora_name]):
                self.lora_start_idx_[lora_name] = 0
                self.lora_cnt_epochs_[lora_name] += 1

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
                                  tokens_len_without_pad_=tokens_without_pad_len_list,
                                  )
