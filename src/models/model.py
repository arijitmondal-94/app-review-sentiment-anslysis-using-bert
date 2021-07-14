import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.config import *
from src.models.sentiment_classifier import SentimentClassifier

class SentimentModel:
    
    def __init__(self):
        self.device = device()
        self.tokenizer = BertTokenizer.from_pretrained(model())
        
        classifier = SentimentClassifier(len(class_names()))
        classifier.load_state_dict(torch.load(trained_model()), map_locations=self.device)
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)
        
    def make_prediction(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=max_length(),
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        
        with torch.no_grad():
            probability = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        
        confidence, predicted_class = torch.max(probability, dim=1)
        predicted_class = predicted_class.cpu().item()
        probability = probability.flatten().cpu().numpy().tolist()
        
        return(
            class_names()[predicted_class],
            confidence,
            dict(class_names(), probability)
        )
        
model = SentimentModel()

def get_model():
    return model
            
        

 
