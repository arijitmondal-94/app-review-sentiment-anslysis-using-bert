import torch
from torch.utils.data import Dataset


class GPReviewDataset(Dataset):
    def __init__(self, review, target, tokenizer, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        return {
            'review_text': review,
            'input_ids' : encoding['input_ids'].flatten(), # flattening to reduce that extra un-necessary dimension
            'attention_mask' : encoding['attention_mask'].flatten(),
            'targets' : torch.tensor(self.target[item], dtype=torch.long) # type long for classification problem
        }