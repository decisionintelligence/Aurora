import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from transformers import BertTokenizer

GLOBAL_TOKENIZER = BertTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aurora", "bert_config"))

def generate_pretrain_dataset(data_dir, seq_len, pred_len):
    time_path = os.path.join(data_dir, 'path_to_your_datasets')
    text_path = os.path.join(data_dir, 'path_to_your_corresponding_text')
    files = os.listdir(time_path)
    datasets = []
    print('Loading Pretrain Datasets...')
    for time_file in tqdm(files):
        try:
            if time_file.endswith('.csv'):
                text_file = os.path.splitext(time_file)[0] + '.json'
            dataset = Aurora_Single_Dataset(os.path.join(time_path, time_file), os.path.join(text_path, text_file),
                                            seq_len, pred_len)
            if len(dataset) > 0:
                datasets.append(dataset)
        except:
            continue

    pretrain_dataset = ConcatDataset(datasets)

    return pretrain_dataset


class Aurora_Single_Dataset(Dataset):

    def __init__(self, root_path, text_root_path, seq_len, pred_len):
        self.root_path = root_path
        self.text_root_path = text_root_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.interval = 2000
        self.max_length = 125

        self.__read_data__()

    def _tokenize(self, texts):
        return GLOBAL_TOKENIZER(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __read_data__(self):
        self.scaler = StandardScaler()
        raw_data = pd.read_csv(self.root_path)
        data = raw_data.drop(columns='date').values.astype('float32')

        scaled_data = self.scaler.fit_transform(data)
        self.data = scaled_data

        with open(self.text_root_path, 'r', encoding='utf-8') as file:
            self.text_list = json.load(file)

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len

        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = self.data[x_begin: x_end]
        seq_y = self.data[y_begin: y_end]

        text_id = index // self.interval
        text = self.text_list[text_id]['text']
        tokenized_text = self._tokenize(text)
        text_input_ids = tokenized_text['input_ids'].squeeze(0)
        text_attention_mask = tokenized_text['attention_mask'].squeeze(0)
        text_token_type_ids = tokenized_text.get('token_type_ids', torch.zeros_like(text_input_ids)).squeeze(0)


        return {
            'input_ids': np.squeeze(seq_x, axis=-1),
            'labels': np.squeeze(seq_y, axis=-1),
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'text_token_type_ids': text_token_type_ids
        }

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def invert_transform(self, data):
        return self.scaler.inverse_transform(data)
