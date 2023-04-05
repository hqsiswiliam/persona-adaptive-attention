import argparse
import os

import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from config_loader.config import extend_compatibility_for_paa_transformer
from dataset.dataset import PersonaChatDataset, get_dataloader
from utils.get_model_by_config import get_model_via_config
from utils.get_tokenizer import get_tokenizer
from utils.save_load_model import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--mdir', type=str, default=None)
parser.add_argument('--max_bz', type=int, default=4)
parser.add_argument('--min_bz', type=int, default=4)
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--model_path', type=str, default='ckpt/paa.pt')
parser.add_argument('--prefix',type=str, default=None)
parser.add_argument('--save_path',type=str, default=None)

from glob import glob

args = parser.parse_args()

MAX_BATCH_SIZE = args.max_bz
MIN_BATCH_SIZE = args.min_bz
if MAX_BATCH_SIZE < MIN_BATCH_SIZE:
    MAX_BATCH_SIZE = MIN_BATCH_SIZE
mdir = args.mdir

result = []
model_paths = glob("{}/*/best.pt".format(mdir))
if args.model_path is not None:
    model_paths = [args.model_path]
for model_path in model_paths:
    skip = False
    for entry in result:
        if model_path in entry:
            print("Skip")
            skip = True
            break
    if skip:
        continue
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    if config.training.batch_size < MIN_BATCH_SIZE:
        config.training.batch_size = MIN_BATCH_SIZE
    if config.training.batch_size > MAX_BATCH_SIZE:
        config.training.batch_size = MAX_BATCH_SIZE
    config = extend_compatibility_for_paa_transformer(config)
    tokenizer = get_tokenizer(config.tokenizer.vocab)
    max_context_turns = config.dataset.max_context_turns
    test_dataset = PersonaChatDataset(config.dataset.test, tokenizer.sep_token, max_context_turns=max_context_turns,
                                      extend_candidates=False)
    test_dataloader = get_dataloader(test_dataset, tokenizer, config, num_workers=args.worker, batch_size_ratio=1)
    model = get_model_via_config(config, tokenizer)
    model.to('cuda')
    load_model(model, path=model_path)
    probs = {}
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    criterion_mean = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')
    all_preds = []
    all_labels = []
    pbar = tqdm(test_dataloader, desc='calculating F1', total=len(test_dataloader))
    for input_data in pbar:
        device = 'cuda'
        model.eval()
        with torch.no_grad():
            logits, _ = model(input_data=input_data,
                              return_loss=True,
                              is_train=True,
                              persona_query_input=input_data['persona_query_input'].to(device),
                              target_input=input_data['target_input'].to(device),
                              )
            labels = input_data['target_input']['input_ids']
            logits = logits[:, :-1, :]
            flatten_logits = logits.argmax(axis=-1).cpu().view(-1).numpy().tolist()
            flatten_labels = labels.cpu().view(-1).numpy().tolist()
            batch_f1 = f1_score(flatten_logits,
                                flatten_labels, average='macro')
            all_preds += flatten_logits
            all_labels += flatten_labels
            current_f1 = f1_score(all_preds,
                                  all_labels, average='macro')
            pbar.set_postfix_str('current F1: {}'.format(current_f1))

    final_f1 = f1_score(all_preds, all_labels, average='macro')
    print("Model Path: {}".format(model_path))
    print("F1: {}".format(final_f1))
    result_str = "Model Path: {}\nF1: {}\n".format(model_path, final_f1)
    result.append(result_str)
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = "generated_text/F1/"
    if args.prefix is not None:
        save_path = "generated_text/{}/F1/".format(args.prefix)
    save_filepath = "{}/{}.txt".format(save_path, config.experiment.name)
    os.makedirs(save_path, exist_ok=True)
    with open(save_filepath, 'w') as file:
        file.write(result_str)
    del model
