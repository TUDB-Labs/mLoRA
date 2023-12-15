from mlora import Tokenizer
from mlora import MultiLoraBatchData
from mlora import LoraBatchDataConfig

import sys
import math
import json
import random
import datasets
from dataclasses import dataclass
from typing import Dict, List, Union


Tokens = List[int]


@dataclass
class TrainData:
    prompt_: str = ""
    tokens_: Tokens = None


@dataclass
class TemplateData:
    parameter_: List[str] = None
    prompt_: str = ""
    prompt_without_input_: str = ""


def load_dataset(data_path: str):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        return datasets.load_dataset("json", data_files=data_path)
    else:
        return datasets.load_dataset(data_path)


class TrainTask():
    tokenizer_: Tokenizer = None

    adapter_name_: str = ""
    data_path_: str = ""
    test_data_path_: str = ""
    prompt_template_path_: str = ""

    # the token list for train and test
    val_set_size: Union[int, float] = -1
    train_token_data_: List[TrainData] = None
    test_token_data_: List[TrainData] = None

    template_data_: TemplateData = None

    # train parameter
    total_epoch_num_: int = -1
    max_train_batch_size_: int = -1
    max_train_micro_batch_size_: int = -1
    max_test_batch_size_: int = -1

    train_cutoff_len_: int = -1
    group_by_length_: bool = False
    expand_side_: str = "left"
    expand_token_id_: int = -1

    # count the stat of train and test data
    epoch_cnt_: int = 1
    next_train_data_start_idx_: int = 0
    next_test_data_start_idx_: int = 0

    def __init__(self,
                 tokenzer: Tokenizer,
                 adapter_name: str,
                 data_path: str,
                 val_set_size: Union[int, float],
                 test_data_path: str,
                 prompt_template_path: str,
                 total_epoch_num: int,
                 max_train_batch_size: int,
                 max_train_micro_batch_size: int,
                 max_test_batch_size: int,
                 train_cutoff_len: int = 256,
                 group_by_length: bool = True,
                 expand_side: str = "right",
                 expand_token_id: int = 0):
        self.tokenizer_ = tokenzer
        self.adapter_name_ = adapter_name
        self.data_path_ = data_path
        self.val_set_size = val_set_size
        self.test_data_path_ = test_data_path
        self.prompt_template_path_ = prompt_template_path
        self.total_epoch_num_ = total_epoch_num
        self.max_train_batch_size_ = max_train_batch_size
        self.max_train_micro_batch_size_ = max_train_micro_batch_size
        self.max_test_batch_size_ = max_test_batch_size
        self.train_cutoff_len_ = train_cutoff_len
        self.group_by_length_ = group_by_length
        self.expand_side_ = expand_side
        self.expand_token_id_ = expand_token_id

    def __load_template_data(self):
        assert self.template_data_ is None
        with open(self.prompt_template_path_, "r", encoding="utf8") as fp:
            template_config_obj = json.load(fp)
        self.template_data_ = TemplateData(
            parameter_=template_config_obj["parameter"],
            prompt_=template_config_obj["prompt"],
            prompt_without_input_=template_config_obj["prompt_no_input"]
        )

    # read from file and replace the template
    def __parse_data_with_template(self,
                                   data: List) -> List[str]:

        ret_data_text: List[str] = []
        for raw_data in data:
            raw_data_obj = {}

            check_without_input_flag = False
            for para in self.template_data_.parameter_:
                if para not in raw_data or raw_data[para] is None:
                    check_without_input_flag = True
                    continue
                raw_data_obj[para] = raw_data[para]

            text_data: str = ""
            if check_without_input_flag:
                text_data = self.template_data_.prompt_without_input_
            else:
                text_data = self.template_data_.prompt_

            for para in self.template_data_.parameter_:
                if para not in raw_data_obj:
                    continue
                text_data = text_data.replace(
                    "{" + para + "}", raw_data[para])

            ret_data_text.append(text_data)

        return ret_data_text

    def __encode_prompt(self,
                        lora_text_data: List[str],
                        is_train_data: bool = True) -> List[TrainData]:
        ret: List[TrainData] = []
        for idx, text in enumerate(lora_text_data):
            if is_train_data:
                tokens = self.tokenizer_.encode(text, bos=True, eos=True)
                if len(tokens) > self.train_cutoff_len_:
                    tokens = tokens[:self.train_cutoff_len_]
            else:
                tokens = self.tokenizer_.encode(text, bos=True, eos=False)

            ret.append(TrainData(prompt_=text, tokens_=tokens))
            if idx % 10000 == 0:
                print(
                    f"encode text data {self.adapter_name_}: {idx}/{len(lora_text_data)}")

        if is_train_data and self.group_by_length_:
            ret.sort(key=lambda x: len(x.tokens_), reverse=True)
        else:
            random.shuffle(ret)

        return ret

    def load_data(self):
        self.__load_template_data()
        data = load_dataset(self.data_path_)
        if self.test_data_path_ is None:
            if self.val_set_size is None or self.val_set_size <= 0:
                self.train_token_data_ = self.__encode_prompt(
                    self.__parse_data_with_template(data["train"]), True)
                self.test_token_data_ = []
            else:
                train_val = data["train"].train_test_split(
                    test_size=self.val_set_size)
                self.train_token_data_ = self.__encode_prompt(
                    self.__parse_data_with_template(train_val["train"]), True)
                self.test_token_data_ = self.__encode_prompt(
                    self.__parse_data_with_template(train_val["test"]), True)
        else:
            train_data = load_dataset(self.test_data_path_)
            self.train_token_data_ = self.__encode_prompt(
                self.__parse_data_with_template(data["train"]), True)
            self.test_token_data_ = self.__encode_prompt(
                self.__parse_data_with_template(train_data["train"]), True)

    def is_train_done(self):
        if self.epoch_cnt_ <= self.total_epoch_num_:
            return False
        return True

    def is_test_done(self):
        if self.next_test_data_start_idx_ < len(self.test_token_data_):
            return False
        return True

    def reset_test_status(self):
        self.next_test_data_start_idx_ = 0

    # reentry function
    def get_train_deta_max_seq_len(self) -> int:
        start_idx = self.next_train_data_start_idx_
        assert start_idx < len(self.train_token_data_)
        # in this strategy must sort
        return len(self.train_token_data_[start_idx].tokens_)

    # non reentry function
    def get_train_data(self) -> List[TrainData]:
        start_idx = self.next_train_data_start_idx_
        end_idx = start_idx + self.max_train_micro_batch_size_

        ret_data = self.train_token_data_[start_idx:end_idx]

        print(f"{self.adapter_name_} train data:")
        print(
            f"    epoch: {self.epoch_cnt_}/{self.total_epoch_num_} \
            step in epoch: {start_idx}/{len(self.train_token_data_)}")

        self.next_train_data_start_idx_ += self.max_train_micro_batch_size_
        if self.next_train_data_start_idx_ >= len(self.train_token_data_):
            self.next_train_data_start_idx_ = 0
            self.epoch_cnt_ += 1

        return ret_data


class Dispatcher():
    config_ = None
    tokenizer_: Tokenizer = None

    # all train task
    ready_train_task_: List[TrainTask] = None
    running_train_task_: List[TrainTask] = None
    done_train_task_: List[TrainTask] = None

    # the number of max candidate training lora model
    # can chose train data from this dataset
    train_lora_candidate_num_: int = 0
    # the number of simultaneously train lora model
    train_lora_simultaneously_num_: int = 0

    strategy_: str = ""

    def __init__(self,
                 config: Dict[str, any],
                 tokenizer: Tokenizer) -> None:
        self.tokenizer_ = tokenizer
        self.config_ = config

        self.ready_train_task_ = []
        self.running_train_task_ = []
        self.done_train_task_ = []

        self.train_lora_candidate_num_ = config["train_lora_candidate_num"]
        self.train_lora_simultaneously_num_ = config["train_lora_simultaneously_num"]
        self.strategy_ = config["train_strategy"]

        # create ready task
        for lora in config["lora"]:
            self.ready_train_task_.append(
                TrainTask(tokenzer=self.tokenizer_,
                          adapter_name=lora["name"],
                          data_path=lora["data"],
                          val_set_size=lora.get("val_set_size", -1),
                          test_data_path=lora.get("test_data", None),
                          prompt_template_path=lora["prompt"],
                          total_epoch_num=lora["num_epochs"],
                          max_train_batch_size=lora["batch_size"],
                          max_train_micro_batch_size=lora["micro_batch_size"],
                          max_test_batch_size=lora["test_batch_size"],
                          train_cutoff_len=config["cutoff_len"],
                          group_by_length=lora.get("group_by_length", True)))

    def optim_dispatch_strategy(self) -> Dict[str, List[TrainData]]:
        task_len = {}
        for idx, task in enumerate(self.running_train_task_):
            task_len[idx] = task.get_train_deta_max_seq_len()
        # sort to get the seq most similar data
        task_len = sorted(task_len.items(), key=lambda x: x[1], reverse=True)
        # find the mini diff
        min_need_pad_len = sys.maxsize
        win_start_idx = 0
        for sidx in range(0, len(task_len) - self.train_lora_simultaneously_num_ + 1):
            win = task_len[sidx:sidx + self.train_lora_simultaneously_num_]
            need_pad_len = 0
            for i in range(1, len(win)):
                # aligin to the max seq len
                need_pad_len += abs(win[i][1] - win[0][1])
            if need_pad_len < min_need_pad_len:
                min_need_pad_len = need_pad_len
                win_start_idx = sidx
        # the result is win_start_idx
        result_win = task_len[win_start_idx:win_start_idx +
                              self.train_lora_simultaneously_num_]
        ret_train_data = {}
        for result_task_len in result_win:
            task_idx = result_task_len[0]
            ret_train_data[self.running_train_task_[
                task_idx].adapter_name_] = self.running_train_task_[task_idx].get_train_data()

        return ret_train_data

    def none_dispatch_strategy(self) -> Dict[str, List[TrainData]]:
        ret_train_data = {}
        cnt = 0
        for task in self.running_train_task_:
            assert not task.is_train_done()
            if cnt >= self.train_lora_simultaneously_num_:
                break
            ret_train_data[task.adapter_name_] = task.get_train_data()
            cnt += 1
        return ret_train_data

    def check_task_done(self) -> bool:
        if len(self.ready_train_task_) == 0 and len(self.running_train_task_) == 0:
            return True
        return False

    def check_test_done(self) -> bool:
        for task in self.running_train_task_:
            if task.is_train_done():
                return False
        return True

    def reset_test_task(self):
        for task in self.running_train_task_:
            task.reset_test_status()

    # ready task -> running task
    def __dispatch_task_in(self):
        assert len(
            self.running_train_task_) <= self.train_lora_candidate_num_
        if len(self.running_train_task_) == self.train_lora_candidate_num_:
            return
        # chose task into running
        while len(self.running_train_task_) < self.train_lora_candidate_num_ and len(self.ready_train_task_) > 0:
            # TODO to dispatch task
            task = self.ready_train_task_.pop(0)
            # to lazy load data
            task.load_data()
            self.running_train_task_.append(task)

    # running task -> done task
    def __dispatch_task_out(self):
        done_task = [
            task for task in self.running_train_task_ if task.is_train_done()]
        self.running_train_task_ = [
            task for task in self.running_train_task_ if not task.is_train_done()]
        self.done_train_task_.extend(done_task)

    def get_test_data(self) -> MultiLoraBatchData:
        pass

    def get_train_data(self) -> MultiLoraBatchData:
        self.__dispatch_task_in()

        # get task train data
        all_train_data: Dict[str, List[TrainData]] = {}
        if self.strategy_ == "none":
            all_train_data = self.none_dispatch_strategy()
        elif self.strategy_ == "optim":
            all_train_data = self.optim_dispatch_strategy()
        else:
            raise "unkown strategy"

        batch_seq_len: int = -1
        # to align batch token data
        for adapter in all_train_data:
            for data in all_train_data[adapter]:
                batch_seq_len = max(batch_seq_len, len(data.tokens_))

        # all prompts and tokens / config
        batch_seq_len = math.ceil(batch_seq_len / 8) * 8
        prompts: List[str] = []
        expand_side: List[str] = []
        batch_tokens: List[Tokens] = []
        tokens_len_without_pad: List[int] = []
        lora_batch_data_config: List[LoraBatchDataConfig] = []

        # batch the all adapter data
        adapter_start_idx: int = 0
        for adapter in all_train_data:
            adapter_end_idx: int = adapter_start_idx + \
                len(all_train_data[adapter])
            for data in all_train_data[adapter]:
                prompts.append(data.prompt_)
                tokens: Tokens = data.tokens_.copy()
                tokens_len_without_pad.append(len(tokens))
                # get the pad token from lora config
                lora_config = None
                for ilora_conf in self.config_["lora"]:
                    if ilora_conf["name"] == adapter:
                        lora_config = ilora_conf
                pad_side = lora_config.get("expand_side", "right")
                assert pad_side == "right" or pad_side == "left"
                # pad the tokens to align
                while len(tokens) < batch_seq_len:
                    if pad_side == "right":
                        tokens.append(self.tokenizer_.pad_id_)
                    else:
                        tokens.insert(0, self.tokenizer_.pad_id_)
                expand_side.append(pad_side)
                batch_tokens.append(tokens)

            lora_batch_data_config.append(LoraBatchDataConfig(adapter_name_=adapter,
                                                              batch_start_idx_=adapter_start_idx,
                                                              batch_end_idx_=adapter_end_idx))
            adapter_start_idx = adapter_end_idx

        self.__dispatch_task_out()

        return MultiLoraBatchData(prompts_=prompts,
                                  lora_batch_data_config_=lora_batch_data_config,
                                  batch_seq_len_=batch_seq_len,
                                  expand_side_=expand_side,
                                  batch_tokens_=batch_tokens,
                                  tokens_len_without_pad_=tokens_len_without_pad)
