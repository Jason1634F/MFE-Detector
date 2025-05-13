import glob
import random
import pandas
from nltk import WordNetLemmatizer
import Func_Tool
import json
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# -*- coding: utf-8 -*-


def load_hc3_data(filepath):
    Data_human_answers = []
    Data_chatgpt_answers = []
    with open(filepath, encoding="utf-8") as file:
        for key, row in enumerate(file):
            data = json.loads(row)
            for x,y in zip(data["human_answers"],data["chatgpt_answers"]):
                if type(x) == str:
                    Data_human_answers.append(x)
                if type(y) == str:
                    Data_chatgpt_answers.append(y)
    return Data_human_answers, Data_chatgpt_answers


def load_faketext(filepath):
    text_data = []
    csv_files = glob.glob(os.path.join(filepath, "*.csv"))
    for csv_file in csv_files:
        csv_data = pandas.read_csv(csv_file)
        for i in range(len(csv_data['text'])):
            text = csv_data['text'][i]
            if type(text) == str:
                text_data.append(text)
    return text_data


def text_cleaning(text, language):
    text = re.sub(r'\[[^\]]*\]', '', text)
    url_pattern = re.compile(r'https?://[^\s\]\[<>]+|www\.[^\s\]\[<>]+')
    text = url_pattern.sub('', text)
    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F700-\U0001F77F"  
                               u"\u2600-\u2B55" 
                               u"\U0001F780-\U0001F7FF"  
                               u"\U0001F800-\U0001F8FF"  
                               u"\U0001F900-\U0001F9FF"  
                               u"\U0001FA00-\U0001FAFF"  
                               u"\U0001FB00-\U0001FBFF"  
                               u"\U0001FC00-\U0001FCFF"  
                               u"\U0001FD00-\U0001FDFF"  
                               u"\U0001FE00-\U0001FEFF"  
                               u"\U0001FF00-\U0001FFEF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"  
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2"  
                               u"\U0001F980-\U0001F9C0"  
                               u"\U0001F3FB-\U0001F3FF" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n+', '', text)
    if language is True:
        text = re.sub(r'\s+', ' ', text)
    else:
        text = re.sub(r'\s+', '', text)
    text = text.strip()
    word_tokens = word_tokenize(text)
    if language is True:
        length = len(word_tokens)
        if len(word_tokens) < 50:
            return None, 0
    else:
        length = len(text)
        if len(text) < 50:
            return None, 0
    if language is True:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('chinese'))
    word_tokens = [token for token in word_tokens if token.lower() not in stop_words]
    word_tokens = [token.lower() for token in word_tokens]
    lemmatizer = WordNetLemmatizer()
    word_tokens = [lemmatizer.lemmatize(token) for token in word_tokens]
    cleaned_text = ' '.join(word_tokens)
    return cleaned_text, length


def optimize_data(raw_data, sample_num, dataset_name, language):
    data = {}.fromkeys(raw_data).keys()
    cleaned_data = []
    sum_length = 0
    for text in data:
        cleaned_text, length = text_cleaning(text, language)
        if cleaned_text is not None:
            sum_length = sum_length + length
            cleaned_data.append(cleaned_text)
    random.seed(0)
    random.shuffle(cleaned_data)
    if sample_num is not None:
        cleaned_data = cleaned_data[:sample_num]
    print(f"{dataset_name} Total number of samples: {len(cleaned_data)}")
    print(f"{dataset_name} Total number of words: {sum_length}")
    return cleaned_data


if __name__ == '__main__':
    hc3_save_folder = Func_Tool.make_dir("./Data for experiment/hc3/")
    faketext_save_folder = Func_Tool.make_dir("./Data for experiment/faketext/")

    Cn_Data_filepath = "./dataset/HC3/Chinese.jsonl"
    En_Data_filepath = "./dataset/HC3/English.jsonl"
    emnnlp_abstract_filepath = "./dataset/Abstract/emnlp_papers.txt"

    faketext_hwt_filepath = "./dataset/DeepfakeTextDetect/human"
    faketext_mgt_filepath = "./dataset/DeepfakeTextDetect/machine"

    sample_num = None

    Cn_Data_human_answers, Cn_Data_chatgpt_answers = load_hc3_data(Cn_Data_filepath)
    En_Data_human_answers, En_Data_chatgpt_answers = load_hc3_data(En_Data_filepath)

    faketext_HWT = load_faketext(faketext_hwt_filepath)
    faketext_MGT = load_faketext(faketext_mgt_filepath)

    Cn_HWT_text = optimize_data(Cn_Data_human_answers,sample_num, "hc3_Cn_HWT_text", False)
    Cn_MGT_text = optimize_data(Cn_Data_chatgpt_answers,sample_num, "hc3_Cn_MGT_text", False)
    En_HWT_text = optimize_data(En_Data_human_answers,sample_num, "hc3_En_HWT_text", True)
    En_MGT_text = optimize_data(En_Data_chatgpt_answers,sample_num, "hc3_En_MGT_text", True)
    faketext_HWT_text = optimize_data(faketext_HWT,sample_num, "faketext_HWT_text", True)
    faketext_MGT_text = optimize_data(faketext_MGT,sample_num, "faketext_MGT_text", True)

    Func_Tool.save_text_result(Cn_HWT_text, "Cn_HWT_text", hc3_save_folder)
    Func_Tool.save_text_result(Cn_MGT_text, "Cn_MGT_text", hc3_save_folder)
    Func_Tool.save_text_result(En_HWT_text, "En_HWT_text", hc3_save_folder)
    Func_Tool.save_text_result(En_MGT_text, "En_MGT_text", hc3_save_folder)
    Func_Tool.save_text_result(faketext_HWT_text, "faketext_HWT_text", faketext_save_folder)
    Func_Tool.save_text_result(faketext_MGT_text, "faketext_MGT_text", faketext_save_folder)
