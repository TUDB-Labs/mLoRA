import aspen

import json
import torch
import torch.optim

with open('config/lora.json', 'r', encoding='utf8') as fp:
    config = json.load(fp)

args = aspen.LlamaModelArgs()
tokenizer = aspen.Tokenizer(config["token_model"])
tokenizer.pad_id_ = 0
args.max_seq_len_ = 4096
args.device = config["device"]
args.vocab_size_ = tokenizer.n_words_
args.pad_id_ = tokenizer.pad_id_
args.n_heads_ = 32
llama_model = aspen.LlamaModel(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_lora_model(llama_model: aspen.LlamaModel):
    for lora_config in config["lora"]:
        aspen.load_random_lora_7b_weight(
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

    data_set = aspen.DataSet(config, tokenizer)
    aspen.load_llama_7b_weight(
        llama_model, config["base_model"], config["device"])
    init_lora_model(llama_model)

    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(llama_model.get_train_paramas(config))

    step_cnt = 0
    while not data_set.check_done():
        optimizer.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss()
        input: aspen.MultiLoraBatchData = data_set.get_batch_data()

        step_cnt += 1

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

        if step_cnt % config["save_step"] == 0:
            aspen.save_lora_model(llama_model, config, f"{step_cnt}")

    aspen.save_lora_model(llama_model, config)
