from transformers import AutoTokenizer, BertTokenizer
from termcolor import colored


def get_tokenizer(vocab):
    tokenizer = AutoTokenizer.from_pretrained(vocab)
    if 'gpt' in tokenizer.name_or_path.lower() or \
        't5' in tokenizer.name_or_path.lower() or \
        'bart' in tokenizer.name_or_path.lower():
        tokenizer.add_tokens(['[SEP]', 'Q:', 'R:', 'P:', '[CLS]', '[BOS]', '[EOS]', '[PAD]'])
        tokenizer.cls_token = '[CLS]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.bos_token = '[BOS]'
        tokenizer.eos_token = '[EOS]'
        tokenizer.pad_token = '[PAD]'
    return tokenizer
