import torch
import re
from functools import reduce
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import pickle
from config_loader.config import get_config
from dataset.dataset_helper import combine_persona_query_response, read_personachat_split, get_chat_by_turns
from copy import deepcopy


def get_cache_path(*args):
    key = "-".join(list([str(i) for i in args]))
    sub_key = re.sub(r"{}".format(os.sep), '-', key)
    cache_path = "cache{}{}".format(os.sep, sub_key)
    return cache_path


def retrieve_cache(*args, **kwargs):
    if kwargs != {}:
        raise ValueError("Not implement!")
    cache_path = get_cache_path(*args)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as file:
            cache_obj = pickle.load(file)
            return cache_obj
    return None


def save_as_cache(cache_object, *args, **kwargs):
    if kwargs != {}:
        raise ValueError("Not implement!")
    cache_path = get_cache_path(*args)
    os.makedirs('cache', exist_ok=True)
    with open(cache_path, 'wb') as file:
        pickle.dump(cache_object, file)
        return cache_object


class PersonaChatDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, persona_query_token, turns=-1, max_context_turns=-1,
                 add_role_indicator=True, add_persona_indicator=True, start_percent=0.0, load_percent=1.0,
                 extend_candidates=False, num_candidates=0):
        self.path = data_path
        self.persona_query_token = persona_query_token
        self.turns_data = retrieve_cache(data_path, persona_query_token, turns, max_context_turns,
                                         add_role_indicator, add_persona_indicator, 'CANDS=YES')
        if self.turns_data is None:
            persona, query, response, their_persona, candidates = read_personachat_split(data_path)
            persona_chat_data = combine_persona_query_response(persona, query, response, candidates)
            turns_list = list(range(1, turns + 1))
            if turns == -1:
                max_turns = reduce(lambda nxt, acc: acc if acc > nxt else nxt,
                                   [v['response_turns'] for v in persona_chat_data.values()])
                turns_list = list(range(1, max_turns + 1))
            turns_data = []
            for turn in reversed(turns_list):
                turns_data += get_chat_by_turns(persona_chat_data, turn, max_context_turns=max_context_turns
                                                , add_role_indicator=add_role_indicator
                                                , add_persona_indicator=add_persona_indicator)
            turns_data.sort(key=lambda x: len(x['input_str']), reverse=True)
            self.turns_data = turns_data
            save_as_cache(turns_data, data_path, persona_query_token, turns, max_context_turns,
                          add_role_indicator, add_persona_indicator, 'CANDS=YES')
        end_index = int(len(self.turns_data) * load_percent)
        start_index = int(len(self.turns_data) * start_percent)
        if load_percent < 1.0 or start_percent > 0.0:
            self.turns_data = self.turns_data[start_index:end_index]
        self.add_id_and_is_candidate_to_turn_data()
        if extend_candidates:
            self.extend_to_candidates(num_candidates)
            self.turns_data = sorted(self.turns_data, key=lambda x: x['ID'])
            print("extended candidates")

    def add_id_and_is_candidate_to_turn_data(self):
        for index in range(len(self.turns_data)):
            self.turns_data[index]['is_candidate'] = False
            self.turns_data[index]['ID'] = index

    def extend_to_candidates(self, num_candidates):
        print("Size Before: {}".format(len(self.turns_data)))
        removed = 0
        for index, turn_data in enumerate(self.turns_data):
            if turn_data['target'] in turn_data['candidates']:
                self.turns_data[index]['candidates'].remove(turn_data['target'])
            if num_candidates > 0:
                self.turns_data[index]['candidates'] = np.random.choice(
                    self.turns_data[index]['candidates'], size=num_candidates, replace=False).tolist()
                removed += 1
        additional_turn_data = []
        for index, turn_data in tqdm(enumerate(self.turns_data), desc='extending candidates', total=len(self.turns_data)):
            for candidate in turn_data['candidates']:
                revised_turn_data = turn_data.copy()
                revised_turn_data['target'] = candidate
                revised_turn_data['is_candidate'] = True
                revised_turn_data['candidates'] = []
                additional_turn_data.append(revised_turn_data)
        self.turns_data += additional_turn_data
        print("Size After: {}".format(len(self.turns_data)))

    def __getitem__(self, idx):
        dialogue_data = self.turns_data[idx]
        persona = "{}".format(dialogue_data['persona'])
        persona_list = ["{}".format(p) for p in dialogue_data['persona_list']]
        return {'persona': persona,
                'query_array': dialogue_data['input'],
                'query': dialogue_data['input_str'],
                'persona_list': persona_list,
                'persona_query': "{} {} {}".format(persona,
                                                   self.persona_query_token,
                                                   dialogue_data['input_str']),
                'target': "{} {}".format(dialogue_data['target'], '[EOS]'),
                'candidates_list': dialogue_data['candidates'],
                'ID': dialogue_data['ID'],
                'is_candidate': dialogue_data['is_candidate']}

    def __len__(self):
        return len(self.turns_data)


def collate_fn(sample_list):
    to_be_flattened = ['persona', 'query', 'target', 'persona_query',
                       'preprocess_texts', 'persona_list',
                       'candidates_list', 'ID', 'is_candidate']
    dont_be_a_tensor = ['persona_list', 'candidates_list', 'ID', 'is_candidate']
    data = {}
    for key in to_be_flattened:
        if key not in sample_list[0].keys():
            continue
        if sample_list[0][key] is None:
            continue
        flatten_samples = [sample[key] for sample in sample_list]
        if flatten_samples[-1].__class__ == str or key in dont_be_a_tensor:
            data[key] = flatten_samples
        else:
            data[key] = torch.tensor(flatten_samples)
    return data


def collate_fn_with_bert_tokenizer(tokenizer, max_length):
    left_tokenizer = deepcopy(tokenizer)
    left_tokenizer.padding_side = 'left'

    def build_collate_fn(sample_list):
        data = collate_fn(sample_list)
        # Because we will discard [CLS] in targets, so we add max_length by 1
        persona_input = left_tokenizer(data['persona'], return_tensors='pt', add_special_tokens=False,
                                       padding=True)
        persona_query_input = left_tokenizer(data['persona_query'], return_tensors='pt', add_special_tokens=False,
                                             padding=True)
        target_input = tokenizer(data['target'], return_tensors='pt', add_special_tokens=False,
                                 padding=True)
        query_input = left_tokenizer(data['query'], return_tensors='pt', add_special_tokens=False,
                                     padding=True)
        persona_list_input_ids = []
        persona_list_attentions = []
        for persona_list_element in data['persona_list']:
            persona_list_output = left_tokenizer(persona_list_element, return_tensors='pt', add_special_tokens=False,
                                                 padding=True)
            persona_list_input_ids.append(persona_list_output['input_ids'])
            persona_list_attentions.append(persona_list_output['attention_mask'])
        data['persona_input'] = persona_input
        data['persona_query_input'] = persona_query_input
        data['target_input'] = target_input
        data['query_input'] = query_input
        data['persona_list_input_ids'] = persona_list_input_ids
        data['persona_list_attentions'] = persona_list_attentions
        return data

    return build_collate_fn


def get_dataloader(dataset, tokenizer, config, shuffle=False, num_workers=None, batch_size_ratio=1):
    batch_size = config.training.batch_size * batch_size_ratio
    if num_workers is None:
        num_workers = batch_size // 4
    return DataLoader(dataset, batch_size=batch_size,
                      collate_fn=collate_fn_with_bert_tokenizer(tokenizer,
                                                                config.dataset.max_length),
                      shuffle=shuffle,
                      num_workers=num_workers)


if __name__ == '__main__':
    config = get_config('config/default.yml')
    # persona, query, response, their_persona = read_personachat_split(config.dataset.train)
    # persona_chat_data = combine_persona_query_response(persona, query, response)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.vocab)
    train_dataset = PersonaChatDataset(config.dataset.train, config.dataset.persona_query_token)
    train_dataset.__getitem__(0)
    dataloader = get_dataloader(train_dataset, tokenizer, config)
