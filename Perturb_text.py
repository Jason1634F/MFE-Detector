import functools
import os
import time
import numpy
import torch
import tqdm
from torch.utils.data import WeightedRandomSampler
import Func_Tool


def get_random_mask(replace_ratio, tokenized_text, device):
    random_mask = torch.rand([len(tokenized_text)], device=device) < replace_ratio
    while True not in random_mask:
        random_mask = torch.rand([len(tokenized_text)], device=device) < replace_ratio
    return random_mask


def perturb_text(original_text, span_length, replace_ratio, mask_model, mask_tokenizer, device):
    tokenized_text = mask_tokenizer.tokenize(original_text)
    # print("tokenized_text = ", tokenized_text)
    buffer_size = 1
    replace_ratio = replace_ratio * (span_length / (span_length + 2 * buffer_size))
    random_mask = get_random_mask(replace_ratio, tokenized_text, device).to(device)

    filled_number = 0
    for idx in range(len(random_mask)):
        if random_mask[idx].item() is True:
            filled_number += 1
            tokenized_text[idx] = '[M]'
    input_ids = torch.tensor([mask_tokenizer.convert_tokens_to_ids(tokenized_text)]).to(device)

    mask_model.eval()
    with torch.no_grad():
        predict_outputs = mask_model(input_ids=input_ids, labels=input_ids)
    predictions = predict_outputs.logits

    for idx in range(len(random_mask)):
        if random_mask[idx].item() is True:
            values,token_ids = torch.topk(predictions[0, idx],k=75)
            weight = torch.nn.functional.normalize(torch.softmax(values,dim=0,dtype=torch.float32),dim=0)
            predict_id = list(WeightedRandomSampler(weight, num_samples=1, replacement=True))[0]
            input_ids[0][idx] = token_ids[predict_id]
    perturbed_text = mask_tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    return perturbed_text


def get_perturb_result(original_texts, dataset_name, span_length, perturbation_times, replace_ratio, mask_model, mask_tokenizer, device):
    torch.manual_seed(0)
    numpy.random.seed(0)
    perturb_result = []

    temp_save_folder = Func_Tool.make_dir(f"./Data for experiment/perturb text/temp/")
    temp_save_folder = Func_Tool.make_dir(os.path.join(temp_save_folder, dataset_name + '/'))

    perturb_function = functools.partial(perturb_text, span_length=span_length, replace_ratio=replace_ratio,
                                         mask_model=mask_model, mask_tokenizer=mask_tokenizer, device=device)

    counter = 0

    for idx in tqdm.tqdm(range(len(original_texts)), desc=f"Perturbing text: {dataset_name}"):
        perturb_texts = []
        while len(perturb_texts) < perturbation_times:
            perturbation = perturb_function(original_texts[idx])
            if perturbation is None:
                continue
            perturb_texts.append(perturb_function(original_texts[idx]))
        perturb_result.append({
            'original_text': original_texts[idx],
            'perturb_texts': perturb_texts
        })

        counter += 1
        if counter % 100 == 0:
            Func_Tool.save_text_result(perturb_result, dataset_name+" "+str(counter), temp_save_folder)

    for idx in range(len(original_texts)):
        assert len(perturb_result[idx]['perturb_texts']) == perturbation_times, f"Text-{idx} got {len(perturb_result[idx]['perturb_texts'])} perturbed samples!"

    return perturb_result


if __name__ == '__main__':
    device = "cuda"
    data_folder = f"./Data for experiment/"
    cache_dir = Func_Tool.make_dir(f"./cache/")
    model_cache_dir = Func_Tool.make_dir(f"./model-cache/")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    save_folder = Func_Tool.make_dir(f"./Data for experiment/perturb text/")

    start_time = time.time()

    en_mask_model_name = "T5-3B"
    en_mask_tokenizer_name = "T5-3B"
    en_mask_model_path = r'./model-cache/models--T5-3B/snapshots/T5-3B-model'
    en_mask_tokenizer_path = r'./model-cache/models--T5-3B/snapshots/T5-3B-tokenizer'
    en_mask_model, en_mask_tokenizer = Func_Tool.load_t5_model(en_mask_model_path, en_mask_tokenizer_path, model_cache_dir, device)

    Cn_HWT_text = Func_Tool.read_json("./Data for experiment/hc3/Cn_HWT_text.json")
    Cn_MGT_text = Func_Tool.read_json("./Data for experiment/hc3/Cn_MGT_text.json")
    En_HWT_text = Func_Tool.read_json("./Data for experiment/hc3/En_HWT_text.json")
    En_MGT_text = Func_Tool.read_json("./Data for experiment/hc3/En_MGT_text.json")
    faketext_HWT_text = Func_Tool.read_json("./Data for experiment/faketext/faketext_HWT_text.json")
    faketext_MGT_text = Func_Tool.read_json("./Data for experiment/faketext/faketext_MGT_text.json")

    perturbation_times = 10
    span_length = 5
    replace_ratio = 0.3


    Cn_HWT_result = get_perturb_result(Cn_HWT_text, "Cn_HC3_HWT_perturb_text", span_length, perturbation_times, replace_ratio, cn_mask_model, cn_mask_tokenizer, device)
    Func_Tool.save_text_result(Cn_HWT_result, "Cn_HC3_HWT_perturb_text", save_folder)

    Cn_MGT_result = get_perturb_result(Cn_MGT_text, "Cn_HC3_MGT_perturb_text", span_length, perturbation_times, replace_ratio, cn_mask_model, cn_mask_tokenizer, device)
    Func_Tool.save_text_result(Cn_MGT_result, "Cn_HC3_MGT_perturb_text", save_folder)

    En_HWT_result = get_perturb_result(En_HWT_text, "En_HC3_HWT_perturb_text", span_length, perturbation_times, replace_ratio, en_mask_model, en_mask_tokenizer, device)
    Func_Tool.save_text_result(En_HWT_result, "En_HC3_HWT_perturb_text", save_folder)

    En_MGT_result = get_perturb_result(En_MGT_text, "En_HC3_MGT_perturb_text", span_length, perturbation_times, replace_ratio, en_mask_model, en_mask_tokenizer, device)
    Func_Tool.save_text_result(En_MGT_result, "En_HC3_MGT_perturb_text", save_folder)

    faketext_HWT_result = get_perturb_result(faketext_HWT_text, "faketext_HWT_perturb_text", span_length, perturbation_times, replace_ratio, en_mask_model, en_mask_tokenizer, device)
    Func_Tool.save_text_result(faketext_HWT_result, "faketext_HWT_perturb_text", save_folder)

    faketext_MGT_result = get_perturb_result(faketext_MGT_text, "faketext_MGT_perturb_text", span_length, perturbation_times, replace_ratio, en_mask_model, en_mask_tokenizer, device)
    Func_Tool.save_text_result(faketext_MGT_result, "faketext_MGT_perturb_text", save_folder)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")

    print(f"Saved perturbation results to {save_folder}")





