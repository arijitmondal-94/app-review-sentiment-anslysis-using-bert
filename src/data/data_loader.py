from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import GPReviewDataset


def create_dataloaders(df, tokenizer, batch_size):
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_val, df_test = train_test_split(df, test_size=0.5, random_state=42)
     
    ds = GPReviewDataset(target = df.sentiemnt.to_numpy(), tokenizer=tokenizer)
    
    return [DataLoader(ds(review=df), batch_size=batch_size, num_workers=4)for df in [df_train, df_val, df_test]]