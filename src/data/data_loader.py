import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from utils import get_tokenizer

class GPReviewDataset(Dataset):
    def __init__(self, review, target, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review = review
        self.target = target
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        
        encoding = tokenizer.encode_plus(
            review,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'input_ids' : encoding['input_ids'],
            'attention_mask' : encoding['attention_mask'],
            'targets' : torch.tensor(self.target[item], dtype=torch.long)
        }

def create_dataloader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        review = df['content'].to_numpy(),
        target = df['sentiemnt'].to_numpy(),
        tokenizer = get_tokenizer(), # TODO made changes //is this evern right thing to do
        max_len = max_len
    )

    return DataLoader( ds, batch_size = batch_size, num_workers = 4 )