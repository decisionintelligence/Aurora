import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')
GLOBAL_TOKENIZER = BertTokenizer.from_pretrained(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aurora/bert_config'), local_files_only=True)


class Dataset_TimeMMD(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'fewshot']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'fewshot': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        self.max_length = 500

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        # if self.args.use_closedllm==0:
        # text_name='Final_Search_'+str(self.args.text_len)
        text_name = 'fact'
        # else:
        #     print("!!!!!!!!!!!!Using output of closed source llm and Bert as encoder!!!!!!!!!!!!!!!")
        #     text_name="Final_Output"
        df_raw = df_raw[['date'] + cols + [self.target] + ['prior_history_avg'] + ['start_date'] + ['end_date']]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        if self.set_type != 3:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.border1 = border1
            self.border2 = border2
        else:
            border1 = int((1 - self.args.few_shot_ratio) * num_train) - self.seq_len
            border2 = num_train
            self.border1 = border1
            self.border2 = border2

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:, -1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]

        self.data_stamp = data_stamp
        self.date = df_raw[['date']][border1:border2].values
        self.start_date = df_raw[['start_date']][border1:border2].values
        self.end_date = df_raw[['end_date']][border1:border2].values
        self.text = df_raw[[text_name]][border1:border2].values
        for i in range(len(self.text)):
            if pd.isnull(self.text[i][0]):
                self.text[i][0] = 'No information available'
        # self.text[0]
        # array(['Available facts are as follows: 1997-09-22: Zanamivir, a neuraminidase inhibitor, has been shown to be safe and effective in treating adults with influenza A or B virus infections when administered directly to the respiratory tract.
        # [Source: pubmed.ncbi.nlm.nih.gov] 1997-09-15: Objective facts about the Pulic Health and FLU situation:In 1997, studies were conducted on the reactivation of antigen-specific CD8+ memory T cells after influenza virus infection, and on
        # the rapid effector function in CD8+ memory T cells. [Source: pubmed.ncbi.nlm.nih.gov]In 1997, the Spanish flu pandemic was discussed in an article, highlighting the high mortality rate among men between 25 and 29 years old. [Source: www.newyorker.com]
        # The 1918 influenza virus genome was studied, and its parts were found to be present in some individuals with a slow influenza infection. [Source: www.ncbi.nlm.nih.gov]Influenza surveillance was discussed as a prevention strategy in the United States.
        # [Source: stacks.cdc.gov]The Bunyaviridae family of viruses was identified as posing an increasing threat to human health. [Source: www.ncbi.nlm.nih.gov] 1997-09-08: The common cold is a viral infection of the upper respiratory system, including the nose
        # , throat, sinuses, eustachian tubes, trachea, larynx, and bronchial tubes. [Source: www.westsignarama.com]; Influenza epidemics have occurred in North America, with outbreaks of the flu happening in the past. [Source: www.arapacana.com] 1997-09-01: Obje
        # ctive facts about the Pulic Health and FLU situation:The amount of circulating virus in naturally infected avian hosts is generally insufficient to infect the mosquito. [Source: wwwnc.cdc.gov]There was a 70 to 80% drop in mortality due to influenza among
        #  senior citizens in a region of the country. [Source: www.scj.go.jp]Evidence of Rickettsia prowazekii infections in the United States exists. [Source: wwwnc.cdc.gov];'], dtype=object)
        # truncate each self.text so that self.text[0] only contain information of one timestamp: 1997-09-22: Zanamivir, a neuraminidase inhibitor, has been shown to be safe and effective in treating adults with influenza A or B virus infections when administered directly to the respiratory tract. [Source: pubmed.ncbi.nlm.nih.gov]
        # timestamp_pattern = r"\\d{4}-\\d{2}-\\d{2}:"
        # timestamp_pattern = r"\d{4}-\d{2}-\d{2}:"

        # # Function to truncate text to one timestamp's information
        # def truncate_to_one_timestamp(text, pattern):

        #     matches = list(re.finditer(pattern, text))  # Find all timestamp positions
        #     if len(matches) > 1:  # If multiple timestamps are found
        #         return text[matches[0].start():matches[1].start()].strip()  # Extract the first section
        #     return text.strip()  # Return the whole text if only one timestamp exists

        # # Apply truncation to each entry in self.text
        # self.text = [truncate_to_one_timestamp(entry[0], timestamp_pattern) for entry in self.text]

        # for each text in self.text, if == nan, then replace it with 'No information available'

    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y

    def get_prior_y_for_imputation(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        # r_begins = s_ends
        # r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        return prior_y

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])

        return x_start_dates, x_end_dates

    def _tokenize(self, texts):
        return GLOBAL_TOKENIZER(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        text = self.text[s_begin:s_end].reshape(-1).tolist()[0]
        tokenized_text = self._tokenize(text)
        text_input_ids = tokenized_text['input_ids'].squeeze(0)
        text_attention_mask = tokenized_text['attention_mask'].squeeze(0)
        text_token_type_ids = tokenized_text.get('token_type_ids', torch.zeros_like(text_input_ids)).squeeze(0)

        return seq_x, seq_y, text_input_ids, text_attention_mask, text_token_type_ids

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
