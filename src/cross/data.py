# coding: utf-8

import os
import sys
import json
import torch
import logging
import pickle

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from cross.commons import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


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


def get_cross_representation(
    sample,
    tokenizer,
    max_ctxt_length,
    max_cand_length,
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

    mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_tokens = (
        left_context + mention_tokens + right_context
    )

    title_tokens = tokenizer.tokenize(title_tokens)
    cand_tokens = tokenizer.tokenize(cand_tokens)
    if title_tokens is not None:
        cand_tokens = title_tokens + [ent_title_token] + cand_tokens

    cand_tokens = cand_tokens[: max_cand_length - 2]

    all_tokens = [cls_token] + context_tokens + [sep_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    padding = [0] * (max_ctxt_length + max_cand_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_ctxt_length + max_cand_length

    return {
        "tokens": all_tokens,
        "ids": input_ids,
    }


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

        ctxt_cates = sample['context_type']
        ctxt_cate_vec = sample['context_type_ids']
        cand_cates = sample['entity_type']
        cand_cate_vec = sample['entity_type_ids']

        ctxt_cate_vec = np.array(ctxt_cate_vec, dtype=np.long)
        cand_cate_vec = np.array(cand_cate_vec, dtype=np.long)
        cate_vec = np.concatenate((ctxt_cate_vec, cand_cate_vec), axis=0)
        
        sample = (left_context, right_context, mention_tokens, name, description)
        tokens = get_cross_representation(
            sample,
            tokenizer,
            max_context_length,
            max_cand_length,
        )

        record = {
            "context": tokens,
            "category": cate_vec,
            "label": label,
        }

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    labels = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    cate_vecs = torch.tensor(
        select_field(processed_samples, "category"), dtype=torch.long,
    )

    tensor_data = TensorDataset(context_vecs, cate_vecs, labels)
    return None, tensor_data


def process_evaluation_data(
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
        
        ctxt_cates = sample['context_type']
        ctxt_cate_vec = sample['context_type_ids']
        cand_cates = sample['entity_type']
        cand_cate_vec = sample['entity_type_ids']

        ctxt_cate_vec = np.array(ctxt_cate_vec, dtype=np.long).reshape(1,-1)
        cand_cate_vec = np.array(cand_cate_vec, dtype=np.long)
        ctxt_cate_vec = np.repeat(ctxt_cate_vec, cand_cate_vec.shape[0], axis=0)
        cate_vec = np.concatenate((ctxt_cate_vec, cand_cate_vec), axis=1)

        cross_tokens = {
            "tokens": [],
            "ids": [],
        }
        for _name, _description in zip(name, description):
            cross_sample = (left_context, right_context, mention_tokens, _name, _description)
            tokens = get_cross_representation(
                cross_sample, 
                tokenizer, 
                max_context_length,
                max_cand_length,
            )
            cross_tokens["tokens"].append(tokens["tokens"])
            cross_tokens["ids"].append(tokens["ids"])

        record = {
            "context": cross_tokens,
            "category": cate_vec,
            "label": label,
        }

        processed_samples.append(record)

    return processed_samples


def process_multi_data(
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

        ctxt_cates = sample['context_type']
        ctxt_cate_vec = sample['context_type_ids']
        cand_cates = sample['entity_type']
        cand_cate_vec = sample['entity_type_ids']

        ctxt_cate_vec = np.array(ctxt_cate_vec, dtype=np.long).reshape(1,-1)
        cand_cate_vec = np.array(cand_cate_vec, dtype=np.long)
        ctxt_cate_vec = np.repeat(ctxt_cate_vec, cand_cate_vec.shape[0], axis=0)
        cate_vec = np.concatenate((ctxt_cate_vec, cand_cate_vec), axis=1)
        
        for i, (_name, _description) in enumerate(zip(name, description)):
            cross_sample = (left_context, right_context, mention_tokens, _name, _description)
            tokens = get_cross_representation(
                cross_sample, 
                tokenizer, 
                max_context_length,
                max_cand_length,
            )

            record = {
                "context": tokens,
                "category": cate_vec[i],
                "label": label[i],
            }

            processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    labels = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    cate_vecs = torch.tensor(
        select_field(processed_samples, "category"), dtype=torch.long,
    )

    tensor_data = TensorDataset(context_vecs, cate_vecs, labels)
    return None, tensor_data