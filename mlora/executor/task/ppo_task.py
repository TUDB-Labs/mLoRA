import copy
import logging
from typing import List, OrderedDict, Tuple, override, Callable, Dict

import torch
import os
from torch.distributions import Categorical
import json
import numpy as np

from mlora.config import PPOTaskConfig
from mlora.executor.context import TRAINCONTEXT_CLASS, TrainTaskContext
from mlora.model.args import LinearInfo, MLoRADataConfig, Tokens
from mlora.model.modules import AdapterModel
from mlora.model.tokenizer import Tokenizer

from .train_task import TrainTask

class PPOTask(TrainTask):

    critic_context_: TrainTaskContext
    actor_context_: TrainTaskContext
    config_: PPOTaskConfig
    loss_fn_critic: Callable
    loss_fn_actor: Callable
    idx: int #generate index
    now_K_epochs: int 
    now_optim_iter_num: int
    adv: torch.Tensor
    td_target: torch.Tensor
    ret_tokens: torch.Tensor
    label_tokens: torch.Tensor
    state_: int # 0: initial stage      1: decision stage       2: update state     3: iteration state

    def __init__(self, config: PPOTaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        self.ret_tokens=[]
        self.label_tokens=[]
        self.state_=0
        self.idx=1
        self.now_K_epochs=0
        self.now_optim_iter_num=0  
        self.adv=torch.zeros(1)
        self.td_target=torch.zeros(1)
        self.perm=torch.zeros(1)

    def prepare(self, linears_info: OrderedDict[str, LinearInfo],tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer

        # prepare the context and the dataset
        # NOTE: how to recover the sort of dataset
        self._pre_dataset()
        self.ppo_pre_context(linears_info)

        LOSS_CLASS={"mse":self.ppo_mse,"adv_loss":self.ppo_adv_loss}
        self.critic_context_.set_loss_fn(LOSS_CLASS[self.config_.loss_type1_])
        self.actor_context_.set_loss_fn(LOSS_CLASS[self.config_.loss_type2_])

    def ppo_pre_context(self, linears_info: OrderedDict[str, LinearInfo]):
        adapter_type_ = self.config_.adapter_.type_
        adapter_type__= self.config_.adapter__.type_
        assert adapter_type_ in TRAINCONTEXT_CLASS
        assert adapter_type_ in TRAINCONTEXT_CLASS

        self.critic_context_ = TRAINCONTEXT_CLASS[adapter_type_](
            self.config_.adapter_, linears_info
        )
        self.actor_context_= TRAINCONTEXT_CLASS[adapter_type__](
            self.config_.adapter__, linears_info
        )

    def ppo_mse(self, data: torch.Tensor,label: torch.Tensor):
        return  (data - label).pow(2).mean()


    def ppo_adv_loss(self, prob: torch.Tensor, old_prob: torch.Tensor,
                     adv: torch.Tensor, a: torch.Tensor,)-> torch.Tensor:
        entropy = Categorical(prob.view(-1,prob.shape[-1])).entropy().mean(dim=-1)
        prob_a = prob.gather(-1, a).squeeze(-1)
        old_prob_a=old_prob.gather(-1,a).squeeze(-1)
        ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a))
        # a/b == exp(log(a)-log(b))

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.config_.clip_rate_, 1 + self.config_.clip_rate_) * adv
        loss1 = -torch.min(surr1, surr2) - self.config_.entropy_coef_ * entropy
        loss1=loss1.view(-1).mean()

        return loss1

    @override
    def adapter_model(self) -> List[AdapterModel]:
        return [self.critic_context_.adapter_model(), self.actor_context_.adapter_model()]

    @override
    def adapter_name(self) -> list[str]:
        return [self.config_.adapter_.name_,self.config_.adapter__.name_]

    @override
    def switch_device(self, device: str):
        self.critic_context_.switch_device(device)
        self.actor_context_.switch_device(device)
        self.adv=self.adv.to(device)
        self.td_target=self.td_target.to(device)
        self.perm=self.perm.to(device)

    @override
    def data(self, start_idx: int, label_start_idx: int) -> Tuple[List[Tokens], List[Tokens], List[MLoRADataConfig]]:

        logging.info(
            f"Task - {self.actor_context_.name_, self.critic_context_.name_} "
            f"epoch: {self.now_epoch_}/{self.config_.num_epochs_} "
            f"iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_} "
            f"state: {self.state_} "
            f"idx: {self.idx} "
            f"now_K_epoch: {self.now_K_epochs} "
            f"now_optim_num: {self.now_optim_iter_num} "
        )

        if self.state_==0:
            data_idx_s = self.now_data_idx_
            data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_
            # get the train raw string
            batch_str = self.prompter_.generate_prompt(self.data_[data_idx_s:data_idx_e])
            batch_strr=[]
            batch_label_chosen=[]
            batch_label_reject=[]

            for str in batch_str:
                s=str.split('_') # delimiter
                batch_strr.append(s[0])
                batch_label_chosen.append(s[1])
                batch_label_reject.append(s[2])
            batch_str=batch_strr

            # convert the string to tokens
            self.ret_tokens=[]
            self.label_tokens=[]
            actor_tokens = list(
                map(
                    lambda raw_str: self.tokenizer_.encode(
                        raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                    ),
                    batch_str,
                )
            )
            label_tokens_chosen = list(
                map(
                    lambda raw_str: self.tokenizer_.encode(
                        raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                    ),
                    batch_label_chosen,
                )
            )
            label_tokens_reject = list(
                map(
                    lambda raw_str: self.tokenizer_.encode(
                        raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                    ),
                    batch_label_reject,
                )
            )

            h=int(len(actor_tokens))
            w=self.config_.generate_num_
            critic_tokens=[[actor_tokens[0][0]]+[0]*w+[actor_tokens[0][-1]] for i in range(h)]

            self.ret_tokens.extend(actor_tokens)
            self.ret_tokens.extend(critic_tokens)
            self.label_tokens.extend(label_tokens_chosen)
            self.label_tokens.extend(label_tokens_reject)

            self.state_=1

        # include critic and actor models' chosen and reject data
        assert len(self.ret_tokens) % 2 == 0
        l1=len(self.ret_tokens)/2 # min_batch_size
        l2=len(self.ret_tokens[0]) # the real actor's len
        l3=len(self.ret_tokens[-1])# the real critic's len
        l1=int(l1)
        l2=int(l2)
        l3=int(l3)
        
        actor_start_idx = start_idx
        actor_end_idx = actor_start_idx + l1

        critic_start_idx = actor_end_idx
        critic_end_idx = critic_start_idx + l1

        label_s_idx=label_start_idx
        label_e_idx=label_s_idx+len(self.label_tokens)

        def select_action(input1: torch.Tensor, deterministic: bool):
            if(self.idx==l3-1): self.state_=2
            if self.state_!=1:
                return

            idx=self.idx
            with torch.no_grad():
                if deterministic:
                    a = torch.argmax(input1[actor_start_idx:actor_end_idx,l2-1],dim=-1)
                else:
                    input_=torch.softmax(input1[actor_start_idx:actor_end_idx,l2-1].view(-1, input1[actor_start_idx:actor_end_idx].shape[-1]),dim=-1)
                    m = Categorical(input_)
                    a = m.sample()
                    a=a.view(l1,-1)

            for i in range(l1):
                self.ret_tokens[i].append(a[i].item())
            for i in range(l1,2*l1):
                self.ret_tokens[i][idx]=a[i-l1].item()
            self.idx+=1

        def loss_fn(
                input1: torch.Tensor, input2: torch.Tensor, batch_tokens:List[Tokens], r: torch.Tensor,
        )-> Tuple[torch.Tensor,bool]:

            if self.state_!=2:
                return

            #Dividing a long trajectory into shorter trajectories for updating
            assert (self.config_.generate_num_)%self.config_.optim_num_==0
            data_len=int(self.config_.generate_num_/self.config_.optim_num_)

            v=torch.tanh(input2[critic_start_idx:critic_end_idx,1:l3].squeeze(dim=-1))
            v_=v.clone().detach()
            v_[:,-1]=0


            p=input1[actor_start_idx:actor_end_idx,l2-l3+1:l2-1]

            batch=torch.tensor(batch_tokens)
            action=batch[critic_start_idx:critic_end_idx,1:-1].unsqueeze(dim=-1)
            action=action.to(self.adv.device)

            if(self.now_K_epochs==0 and self.now_optim_iter_num==0):
                deltas=torch.zeros_like(v_)
                for j in range(1,len(deltas[0])):
                    deltas[:,j-1]=r[:,j-1]+self.config_.gamma_*v_[:,j]-v_[:,j-1]

                adv=torch.zeros_like(v_)

                for j in range(len(adv[0])-2,-1,-1):
                    adv[:,j]=deltas[:,j]+self.config_.gamma_*self.config_.lamdb_*adv[:,j+1]
                
                adv=torch.flip(adv,[-1])
                adv = adv[:,0:-1]
                v_=v_[:,0:-1]
                td_target = adv + v_
                self.adv=adv
                self.td_target=td_target
                self.old_p=torch.softmax(p,dim=-1)

            if(self.now_optim_iter_num==0):
                self.perm = torch.randperm(len(self.adv[0])) 

            adv_ = self.adv[:,self.perm].clone().detach()
            td_target_ = self.td_target[:,self.perm].clone().detach()
            old_p = self.old_p[:,self.perm].clone().detach()
            p=torch.softmax(p,dim=-1)
            p = p[:,self.perm]
            v_=v_[:,self.perm]
            action=action.to(self.adv.device)
            action=action[:,self.perm]

            index=[i for i in range(self.now_optim_iter_num * data_len, min((self.now_optim_iter_num + 1) * data_len, len(self.adv[0])))]
            loss1=self.critic_context_.loss_fn_(v_[:,index].view(-1),td_target_[:,index].view(-1))
            loss2=self.actor_context_.loss_fn_(p[:,index],old_p[:,index],adv_[:,index],action[:,index])
            loss=loss1+loss2

            self.now_optim_iter_num+=1
            if(self.now_optim_iter_num==self.config_.optim_num_):
                self.now_K_epochs+=1
                self.now_optim_iter_num=0

            if(self.now_K_epochs==self.config_.K_epochs_): 
                self.now_K_epochs=0
                self.state_=3

            logging.info(f"Adapter {self.critic_context_.name_} loss: {loss1} ")
            logging.info(f"Adapter {self.actor_context_.name_} loss: {loss2} ")

            return loss

        def work(input1: torch.Tensor, input2: torch.Tensor, deterministic: bool, batch_tokens, labels,
        r: torch.tensor, beta=1.0):
            #input1: policy     input2: critic      
            select_action(input1,False)
            loss=None
            loss=loss_fn(input1,input2,batch_tokens,r)
            return loss
        
        critic_start_idx=int(critic_start_idx)
        critic_end_idx=int(critic_end_idx)
        actor_start_idx=int(actor_start_idx)
        actor_end_idx=int(actor_end_idx)

        actor_data_config = MLoRADataConfig(
            self.actor_context_.name_,
            self.actor_context_.type_,
            actor_start_idx,
            actor_end_idx,
            self._expand_batch_tokens,
            lambda *_: None,
            self.task_name(),
            label_start_idx=label_s_idx,
            label_end_idx=label_e_idx,
        )
        critic_data_config = MLoRADataConfig(
            self.critic_context_.name_,
            self.critic_context_.type_,
            critic_start_idx,
            critic_end_idx,
            self._expand_batch_tokens,
            work,
            self.task_name(),
            task_type=self.task_type(),
            label_start_idx=label_s_idx,
            label_end_idx=label_e_idx,
        )

        return self.ret_tokens, self.label_tokens, [actor_data_config,critic_data_config]

    @override
    def step(self):
        if self.state_==0 or self.state_==1:
            return

        stepd: bool = False
        need_checkpoint: bool = False

        if self.now_step_ % self.config_.accumulate_step_ == 0:
            stepd = True
            self.critic_context_.step()
            self.actor_context_.step()

        if self.now_step_ % self.config_.save_step_ == 0:
            need_checkpoint = True

        self.now_step_ += 1

        if self.state_==3: 
            self.now_data_idx_ += self.config_.mini_batch_size_
            self.state_=0
            self.idx=1

        if self.now_data_idx_ >= len(self.data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # to save the checkpoint, must ensure the order
        # beacuse we need recover the state
        if need_checkpoint:
            self._save(is_checkpoint=True)

        # task finish we also need to step
        if not stepd and self.now_epoch_ >= self.config_.num_epochs_:
            self.actor_context_.step()
            self.critic_context_.step()

    def _save(self, is_checkpoint: bool = False, additional_info: Dict[str, str] = {}):
        output_dir1 = self.critic_context_.path_
        output_dir2 = self.actor_context_.path_

        if is_checkpoint:
            checkpoint_folder = "checkpoint_" + "_".join(
                [
                    str(self.now_step_),
                    str(self.now_epoch_),
                    str(self.now_data_idx_),
                ]
            )
            output_dir1 = self.critic_context_.path_ + os.sep + checkpoint_folder
            output_dir2 = self.actor_context_.path_ + os.sep + checkpoint_folder

        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)


        # save to disk, if save checkpoint, we need also save the state dict
        if is_checkpoint:
            torch.save(
                {
                    "weight_dict": self.critic_context_.weight_dict(),
                    "state_dict": self.critic_context_.state_dict(),
                },
                output_dir1 + os.sep + "checkpoint.bin",
            )
            torch.save(
                {
                    "weight_dict": self.actor_context_.weight_dict(),
                    "state_dict": self.actor_context_.state_dict(),
                },
                output_dir2 + os.sep + "checkpoint.bin",
            )
            # Save checkpoint for shuffle_data.
            self._save_data(output_dir1)
            self._save_data(output_dir2)
        else:
            torch.save(
                self.critic_context_.weight_dict(), output_dir1 + os.sep + "adapter_model.bin"
            )
            torch.save(
                self.actor_context_.weight_dict(), output_dir2 + os.sep + "adapter_model.bin"
            )


        adapter_config: Dict[str, str] = {}
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}
        adapter_config = {**adapter_config, **self.config_.adapter_.export()}

        with open(output_dir1 + os.sep + "adapter_config.json", "w") as f1:
            json.dump(adapter_config, f1, indent=4)
        with open(output_dir2 + os.sep + "adapter_config.json", "w") as f2:
            json.dump(adapter_config, f2, indent=4)

    @override
    def done(self):
        self._save(is_checkpoint=False)
        # Delete the cache file.
        self._del_cache_file()
        # release the context
        del self.critic_context_
        del self.actor_context_

    @override
    def terminate(self):
        del self.critic_context_
        del self.actor_context_