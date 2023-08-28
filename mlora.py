import json
import torch
from aspen import LlamaModel, Tokenizer, DataSet
from aspen import LlamaModelArgs, MultiLoraBatchData
from aspen import load_llama_7b_weight, load_random_lora_7b_weight
from aspen import save_lora_model
import torch.optim

with open('config/lora.json', 'r', encoding='utf8') as fp:
    config = json.load(fp)

args = LlamaModelArgs()
tokenizer = Tokenizer(config["token_model"])
tokenizer.pad_id_ = 0
args.max_seq_len_ = 4096
args.device = config["device"]
args.vocab_size_ = tokenizer.n_words_
args.pad_id_ = tokenizer.pad_id_
args.n_heads_ = 32
llama_model = LlamaModel(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_lora_model(llama_model: LlamaModel):
    for lora_config in config["lora"]:
        load_random_lora_7b_weight(
            llama_model,
            lora_config["name"],
            lora_config["r"],
            llama_model.dim_,
            lora_config["target_modules"],
            llama_model.device_)
        llama_model.update_lora_configure(
            lora_config["name"], lora_config["r"], lora_config["alpha"], lora_config["dropout"])


if __name__ == "__main__":
    setup_seed(42)

    data_set = DataSet(config, tokenizer)
    load_llama_7b_weight(llama_model, config["base_model"], config["device"])
    init_lora_model(llama_model)

    torch.cuda.empty_cache()

    # optim begin
    optimizer = torch.optim.SGD(
        llama_model.get_train_paramas(config), lr=1e-3)
    # optim end

    step = 0
    # torch.autograd.set_detect_anomaly(True)
    while not data_set.check_done():
        optimizer.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss()
        input: MultiLoraBatchData = data_set.get_batch_data()

        step += 1

        output = llama_model.forward(input)
        labels = torch.tensor(input.batch_tokens_,
                              dtype=torch.long).to(config["device"])

        total_loss = None
        for lora_config in input.lora_batch_data_config_:
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss_input = output[start_idx:end_idx][..., :-1,
                                                   :].contiguous().view(-1, llama_model.vocab_size_)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1)
            loss = loss_fn(loss_input, loss_target)
            print(
                f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        optimizer.step()

        if step % 200 == 0:
            for lora_config in config["lora"]:
                save_lora_model(
                    llama_model, lora_config["output"] + f".chk{step}", lora_config["name"])

    for lora_config in config["lora"]:
        save_lora_model(
            llama_model, lora_config["output"], lora_config["name"])
