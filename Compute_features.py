import time
import Func_Tool
import os
import numpy
import torch
import tqdm
from transformers import logging

logging.set_verbosity_error()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# -*- coding: utf-8 -*-
def compute_features(text, tokenizer, model, device):
    torch.manual_seed(0)
    numpy.random.seed(0)
    with torch.no_grad():
        tokenized = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        label = tokenized.input_ids.to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        text_len = torch.sum(attention_mask, dim=1).item()
        if text_len < 50:
            return None

        output = model(input_ids=label, attention_mask=attention_mask, labels=label)


        logit = output.logits.to(device)
        neg_entropy = (torch.nn.functional.softmax(logit, dim=-1) * torch.nn.functional.log_softmax(logit, dim=-1)).to(device)
        entropy = -neg_entropy.sum(-1).mean().item()
        matche = (logit.argsort(-1, descending=True) == label.unsqueeze(-1)).nonzero().to(device)

        assert matche.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matche.shape}"

        rank, timestep = matche[:, -1].to(device), matche[:, -2].to(device)

        assert (timestep == torch.arange(len(timestep)).to(timestep.device)).all(), "Expected one match per timestep"

        log_rank = torch.log(rank.float() + 1).mean().item()
        log_rank_var = torch.log(rank.float() + 1).var().item()
        rank = (rank.float() + 1).mean().item()

        likelihood = torch.nn.functional.log_softmax(logit, dim=-1).to(device)

        likelihood_list = likelihood.gather(dim=-1, index=label.unsqueeze(-1)).squeeze(-1).to(device)

        surprisal_list = -likelihood_list.to(device)
        surprisal_mean = surprisal_list.mean().item()
        surprisal_var = surprisal_list.var().item()
        surprisal_token_var = (surprisal_list[0,1:]-surprisal_list[0,:-1]).pow(2).mean(dtype=torch.float16).item()

        likelihood = likelihood_list.mean().item()

        PPL = torch.exp(-likelihood_list.sum()/text_len).item()

        unique_label = torch.unique(label).to(device)
        vocabulary = unique_label.size()[0]
        density = vocabulary / text_len

        rank_likelihood_ratio = -rank/likelihood

        neg_likelihood = surprisal_mean

    return [entropy, rank, log_rank, log_rank_var, rank_likelihood_ratio, neg_likelihood, surprisal_var, surprisal_token_var, PPL, density]


def supervised_detect(text, tokenizer, model, device, text_category_label):
    with torch.no_grad():
        tokenized = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
        text_len = torch.sum(tokenized['attention_mask'], dim=1).item()
        if text_len < 50:
            return None
        output = model(**tokenized)
        logits = output.logits
        text_pred = logits.softmax(-1)[:, 0].item()
    return [text_pred, text_category_label]


def get_features(data, features, name, text_category_label, language, eval_model, eval_tokenizer, detect_model, detect_tokenizer, device):
    for text in tqdm.tqdm(data, total=len(data), desc=f"Computing features: " + name):
        feature_result = compute_features(text, eval_tokenizer, eval_model, device)
        if feature_result is not None:
            detect_result = supervised_detect(text, detect_tokenizer, detect_model, device, text_category_label)
            if detect_result is not None:
                if language is True:
                    detect_result[0] = 1 - detect_result[0]
                feature_result.extend(detect_result)
                features.append(feature_result)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    device = "cuda"
    data_folder = f"./Data for experiment/"
    cache_dir = Func_Tool.make_dir(f"./cache/")
    model_cache_dir = Func_Tool.make_dir(f"./model-cache/")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    save_folder_features = Func_Tool.make_dir(f"./result/features/")

    '''
    eval_model_name = "gpt2-medium"
    eval_model_path = r'./model-cache/models--gpt2-medium/snapshots/gpt2-medium'
    eval_tokenizer_path = r'./model-cache/models--gpt2-medium/snapshots/gpt2-medium'
    eval_model, eval_tokenizer = Func_Tool.load_gpt_model(eval_model_path, eval_tokenizer_path, model_cache_dir, device)
    '''
    eval_model_name = "gpt2-xl"
    eval_model_path = r"./model-cache/models--gpt2-xl/snapshots/gpt2-xl-model"
    eval_tokenizer_path = r"./model-cache/models--gpt2-xl/snapshots/gpt2-xl-tokenizer"
    eval_model, eval_tokenizer = Func_Tool.load_gpt_model(eval_model_path, eval_tokenizer_path, model_cache_dir, device)

    En_supervised_model_name = 'roberta-large-openai-detector'
    En_supervised_model_path = r'./model-cache/models--roberta-large-openai-detector/snapshots/detector_model'
    En_supervised_tokenizer_path = r'./model-cache/models--roberta-large-openai-detector/snapshots/detector_tokenizer'
    En_detect_model, En_detect_tokenizer = Func_Tool.load_detect_model(En_supervised_model_path, En_supervised_tokenizer_path, model_cache_dir, device)

    Cn_supervised_model_name = "Juner/AI-generated-text-detection-pair"
    Cn_supervised_model_path = r'./model-cache/models--Juner--AI-generated-text-detection-pair/snapshots/Juner_model'
    Cn_supervised_tokenizer_path = r'model-cache/models--Juner--AI-generated-text-detection-pair/snapshots/Juner_tokenizer'
    Cn_detect_model, Cn_detect_tokenizer = Func_Tool.load_detect_model(Cn_supervised_model_path, Cn_supervised_tokenizer_path, model_cache_dir, device)

    eval_model.to(device)
    En_detect_model.to(device)
    Cn_detect_model.to(device)

    Cn_HWT_text = Func_Tool.read_json("./Data for experiment/hc3/Cn_HWT_text.json")
    Cn_MGT_text = Func_Tool.read_json("./Data for experiment/hc3/Cn_MGT_text.json")
    En_HWT_text = Func_Tool.read_json("./Data for experiment/hc3/En_HWT_text.json")
    En_MGT_text = Func_Tool.read_json("./Data for experiment/hc3/En_MGT_text.json")

    faketext_HWT_text = Func_Tool.read_json("./Data for experiment/faketext/faketext_HWT_text.json")
    faketext_MGT_text = Func_Tool.read_json("./Data for experiment/faketext/faketext_MGT_text.json")


    title = ['entropy', 'rank', 'log_rank', "log_rank_var", 'rank_likelihood_ratio', "neg_likelihood", "surprisal_var", "surprisal_token_var", 'PPL', 'density', 'supervised_label', 'label']
    Cn_features = [title]
    En_features = [title]
    faketext_features = [title]


    start_time = time.time()

    get_features(Cn_HWT_text, Cn_features, "Cn_HWT", 0, True, eval_model, eval_tokenizer, Cn_detect_model, Cn_detect_tokenizer, device)
    get_features(Cn_MGT_text, Cn_features, "Cn_MGT", 1, True, eval_model, eval_tokenizer, Cn_detect_model, Cn_detect_tokenizer, device)
    Func_Tool.save_data_features(Cn_features, "Cn_features", save_folder_features)

    get_features(En_HWT_text, En_features, "En_HWT", 0, False, eval_model, eval_tokenizer, En_detect_model, En_detect_tokenizer, device)
    get_features(En_MGT_text, En_features, "En_MGT", 1, False, eval_model, eval_tokenizer, En_detect_model, En_detect_tokenizer, device)
    Func_Tool.save_data_features(En_features, "En_features", save_folder_features)

    get_features(faketext_HWT_text, faketext_features, "faketext_HWT", 0, False, eval_model, eval_tokenizer, En_detect_model, En_detect_tokenizer, device)
    Func_Tool.save_data_features(faketext_features, "faketext_features", save_folder_features)
    get_features(faketext_MGT_text, faketext_features, "faketext_MGT", 1, False, eval_model, eval_tokenizer, En_detect_model, En_detect_tokenizer, device)
    Func_Tool.save_data_features(faketext_features, "faketext_features", save_folder_features)

    end_time = time.time()
    print("Run time:", end_time - start_time)
