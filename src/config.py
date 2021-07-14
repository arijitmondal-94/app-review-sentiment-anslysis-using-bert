import toml
import torch

    
config = config = toml.load('pyproject.toml')

def model():
    return config.get('training').get('model')

def tokenizer():
    return config.get('training').get('tokenizer')

def max_length():
    return config.get('training').get('max_length')

def device():
    return torch.cuda.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def epochs():
    return config.get('training').get('epochs')

def batch_size():
    return config.get('training').get('batch_size')

def trained_model():
    return config.get('prediction').get('model')

def class_names():
    return config.get('prediction').get('class_names')
