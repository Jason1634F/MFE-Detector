import json
import os
import csv
import torch
import transformers


def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def read_json(file_path):
    with open(file_path, encoding='utf-8') as context:
        return json.load(context)


def save_text_result(data, name, file_path):
    with open(os.path.join(file_path, f"{name}.json"), "w", encoding='utf-8') as context:
        json.dump(data, context, ensure_ascii=False, indent=4)


def save_data_features(data, name, file_path):
    with open(os.path.join(file_path, f"{name}.csv"), "w", newline='',encoding='utf-8') as context:
        writer = csv.writer(context)
        writer.writerows(data)


def write_txt_data(data, name, file_path):
    with open(os.path.join(file_path, f"{name}.txt"), 'w', encoding='utf-8') as file:
        file.write(data)


def load_gpt_model(model_name_or_path, tokenizer_name_or_path, model_cache_dir, device):
    model_kwargs = {}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_kwargs,
        cache_dir=model_cache_dir).to(device)
    tokenizer_kwargs = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        **tokenizer_kwargs,
        cache_dir=model_cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_detect_model(model_name_or_path, tokenizer_name_or_path, model_cache_dir, device):
    model_kwargs = {}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_kwargs,
        cache_dir=model_cache_dir).to(device)

    tokenizer_kwargs = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        **tokenizer_kwargs,
        cache_dir=model_cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_roberta_model(model_name_or_path, tokenizer_name_or_path, model_cache_dir, device):
    model_kwargs = {}
    model = transformers.RobertaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_kwargs,
        is_decoder=True,
        cache_dir=model_cache_dir).to(device)

    tokenizer_kwargs = {}
    tokenizer = transformers.RobertaTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        **tokenizer_kwargs,
        cache_dir=model_cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_t5_model(model_name_or_path, tokenizer_name_or_path, model_cache_dir, device):
    model_kwargs = dict(torch_dtype=torch.bfloat16)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_kwargs,
        cache_dir=model_cache_dir).to(device)
    tokenizer_kwargs = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        **tokenizer_kwargs,
        cache_dir=model_cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_mamba_model(model_name_or_path, tokenizer_name_or_path, model_cache_dir, device):
    model_kwargs = dict(torch_dtype=torch.bfloat16)
    model = transformers.MambaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_kwargs,
        cache_dir=model_cache_dir).to(device)
    tokenizer_kwargs = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        **tokenizer_kwargs,
        cache_dir=model_cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer