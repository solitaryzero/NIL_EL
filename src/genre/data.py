# coding: utf-8

import os
import sys
import json
import torch
import logging
import pickle

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset
from transformers import BertTokenizer
from genre.commons import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def read_dataset(split_name, data_folder, ratio=1):

    file_name = '%s.jsonl' %split_name
    json_file_path = os.path.join(data_folder, file_name)

    samples = []
    with open(json_file_path, 'rb') as fin:
        for line in fin:
            js = json.loads(line)
            samples.append(js)

    n = len(samples)
    samples = samples[:int(n*ratio)]

    return samples, len(samples)


def read_evaluation_dataset(split_name, data_folder):
    file_name = '%s.jsonl' %split_name
    json_file_path = os.path.join(data_folder, file_name)

    samples = []
    with open(json_file_path, 'rb') as fin:
        for line in fin:
            js = json.loads(line)
            samples.append(js)

    return samples, len(samples)


def read_finetune_dataset(pkl_file_path):
    with open(pkl_file_path, 'rb') as fin:
        all_finetune_data = pickle.load(fin)
        all_context_ids = torch.stack([t['context_id'] for t in all_finetune_data])
        all_entity_index = torch.tensor([t['entity_index'] for t in all_finetune_data], dtype=torch.long)
        all_labels = torch.tensor([t['label'] for t in all_finetune_data], dtype=torch.float)

    return all_context_ids, all_entity_index, all_labels


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_genre_input(
    sample,
    tokenizer,
    max_ctxt_length,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    ent_title_token=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    context_window = max_ctxt_length - 4 # reserve place for ent tags, cls and sep

    left_context, right_context, mention_tokens, title_tokens, cand_tokens = sample
    mention_tokens = tokenizer.tokenize(mention_tokens)
    context_length = context_window-len(mention_tokens)

    left_tokens = tokenizer.tokenize(left_context)
    left_length = min(context_length//2, len(left_tokens))
    right_tokens = tokenizer.tokenize(right_context)
    right_length = min(context_length - left_length, len(right_tokens))
    left_context = left_tokens[-left_length:]
    right_context = right_tokens[:right_length]

    mention_tokens = ent_start_token + ' ' + tokenizer.convert_tokens_to_string(mention_tokens) + ' ' + ent_end_token

    input_seq = tokenizer.convert_tokens_to_string(left_context) + mention_tokens + tokenizer.convert_tokens_to_string(right_context)

    return input_seq


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        left_context = sample['left_context']
        right_context = sample['right_context']
        mention_tokens = sample['mention']
        name = sample['entity_title']
        description = sample['entity_abstract']
        label = sample['label']
        
        sample = (left_context, right_context, mention_tokens, name, description)
        input_seq = get_genre_input(
            sample,
            tokenizer,
            max_context_length,
        )

        answer = 'NIL'
        for i, x in enumerate(label):
            if (x == 1):
                answer = name[i]

        record = {
            "context": input_seq,
            "label": answer,
        }

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + sample["context"])
            logger.info("Answer tokens : " + sample["label"])

        input()

    return processed_samples


class GenreDataset(Dataset):
    def __init__(self, samples):
        super(GenreDataset).__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (sample["context"], sample["label"])