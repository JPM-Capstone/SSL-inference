import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

EOS_token = '2' # RoBERTa

class LabeledDataset(Dataset):

    def __init__(self):

        self.data = pd.read_csv(os.path.join("data", "yahoo_test.csv"), 
                                index_col=0)
                
    def __len__(self):

        return self.data.shape[0]
    
    def __getitem__(self, index):

        input_ids, attention_mask, label = self.data.iloc[index][['input_ids', 'attention_mask', 'labels']]

        input_ids = torch.tensor(np.array(input_ids.split()[1:-1] + [EOS_token], dtype=np.int64)) # convert string to array

        attention_mask = torch.tensor(np.array(['1'] + attention_mask.split()[1:-1] + ['1'], dtype=np.int64))

        return input_ids, attention_mask, label