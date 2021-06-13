import transformers

def get_tokenizer():

    return transformers.BertTokenizer.from_pretrained('bert-base-cased')