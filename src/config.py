import toml
import torch

    
config = config = toml.load('pyproject.toml')

def model():
    return config.get('training-parameters').get('model')

def tokenizer():
    return config.get('training-parameters').get('tokenizer')

def device():
    return torch.cuda.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def epochs():
    return config.get('training-parameters').get('epochs')

def batch_size():
    return config.get('training-parameters').get('batch_size')
