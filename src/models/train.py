import torch
import pandas as pd
import transformers
import numpy as np

from collections import defaultdict
from datetime import datetime
from torch import nn
from transformers import optimization
from src.models.sentiment_classifier import SentimentClassifier
from src.data.data_loader import create_dataloaders
from src.config import *
import src.config as cfg # TODO: fix this


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        target = d['target'].to(device)
        
        output = model(input_ids, attention_mask)
        
        _, prediction = torch.max(output, dim=1)
        loss = loss_fn(output, target)
        
        correct_predictions += torch.sum(prediction == target)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        return correct_predictions.double() / n_examples, np.mean(losses)
    
def eval_model(model, data_loader, loss_fn, device, n_examples):
    
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            target = d['target'].to(device)
            
            output = model(input_ids, attention_mask)
            
            _, prediction = torch.max(output, dim=1)
            loss = loss_fn(output, target)
            
            correct_predictions += torch.sum(prediction == target)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)
            
def main():    
    tokenizer = transformers.BertTokenizer.from_pretrained(cfg.tokenizer())
    train_loader, validation_loader, test_loader =  create_dataloaders(pd.read_csv('data/app_review.csv'), tokenizer, 
                                                                       batch_size())
    model = SentimentClassifier(3)
    total_steps = create_dataloaders(train_loader) * epochs()
    loss_fn = nn.CrossEntropyLoss().to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, correct_bias=False),
    scheduler = optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    history = defaultdict(list)
    best_accuracy = 0
    
    for epoch in range(epochs()):
        print(f'Epoch {epoch + 1}/{epochs()}')
        print('-' * 10)
        
        train_accuracy, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device(), scheduler, 
                                                 len(df_train))
        print(f'Train loss {train_loss} accuracy {train_accuracy}')
        
        val_accuracy, val_loss = eval_model(model, validation_loader, loss_fn, device(), len(df_val))
        print(f'Validation loss {val_loss} accuracy {val_accuracy}')
        print()
        
        history['train_accuracy'].append(train_accuracy)
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)
        
        if val_accuracy > best_accuracy:
            torch.model(model, './app-review-sentiment-analysis-using-bert/data/'+str(datetime.now())+'.pt')

if __name__ == "__main__":
    main()