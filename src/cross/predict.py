# coding: utf-8

import os
import argparse
import pickle
import json
import sys
import random
import time
import numpy as np

from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.multiprocessing as mp
import torch.distributed as dist

from cross.model import CrossModel, get_model_obj, load_model
import cross.data as data
import cross.utils as utils
from cross.params import ClinkParser
import cross.commons as commons


def predict(out, threshold=0.5):
    p = -1
    max_prob = 0.0
    for i, x in enumerate(out):
        if (x >= threshold) and (x > max_prob):
            max_prob = x
            p = i

    return max_prob, p


def evaluate(
    args,
    all_records,
    samples,
    ranker,
    _lambda=0.5,
    _threshold=0.5,
    debug=False,
):
    with torch.no_grad():
        ranker.eval()
        iter_ = tqdm(all_records, desc="Evaluation")
        device = ranker.device

        results = []
    
        for step, record in enumerate(iter_):
            context_ids = torch.tensor(record["context"]["ids"], dtype=torch.long).to(device)
            labels = torch.tensor(record["label"], dtype=torch.float).unsqueeze(0).to(device)

            semantic_scores, category_scores, category_vec, category_ctxt, category_cand = get_model_obj(ranker).score(
                context_input=context_ids, 
            )
            semantic_scores = torch.sigmoid(semantic_scores)
            # logits = semantic_scores

            if (args.score_function == 'add'):
                logits = _lambda*semantic_scores+(1-_lambda)*category_scores
            else:
                logits = semantic_scores*category_scores

            logits = logits.detach().cpu().numpy()
            labels = labels.squeeze(0).detach().cpu().numpy()

            max_prob, predicted_index = predict(logits, threshold=_threshold)
            input_json = samples[step]
            if (predicted_index == -1):
                prediction = 'NIL'
            else:
                prediction = input_json['entity_title'][predicted_index]

            answer = 'NIL'
            for i, x in enumerate(input_json['label']):
                if (x == 1):
                    answer = input_json['entity_title'][i]
                    break
            
            predicted_json = {
                'mention': input_json['mention'],
                'left_context': input_json['left_context'],
                'right_context': input_json['right_context'],
                'entity_title': input_json['entity_title'],
                'prediction': prediction,
                'answer': answer,
                'entity_type': input_json['entity_type'],
            }
            results.append(predicted_json)

        return results


def main(args):
    # load data
    test_samples, test_num = data.read_evaluation_dataset("test", args.evaluation_data_path)

    # load model
    path_to_config = os.path.join(args.model_path, commons.config_name)
    path_to_model = os.path.join(args.model_path, commons.checkpoint_name)

    with open(path_to_config, 'r', encoding='utf-8') as fin:
        params = json.load(fin)
        params["output_path"] = args.output_path
        params['path_to_model'] = path_to_model

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ranker = CrossModel(params)
    tokenizer = ranker.tokenizer
    device = ranker.device
    n_gpu = ranker.n_gpu

    ranker = ranker.to(ranker.device)

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)    

    # build data loader
    test_data = data.process_evaluation_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        debug=params["debug"],
    )

    # evaluation
    results = evaluate(
        args=args,
        all_records=test_data,
        samples=test_samples,
        ranker=ranker,
        _lambda=args.lambd,
        _threshold=args.threshold,
        debug=args.debug,
    )

    result_path = os.path.join(args.output_path, '%s_prediction.txt' %(args.dataset))
    with open(result_path, 'w', encoding='utf-8') as fout:
        for js in results:
            json.dump(js, fout)
            fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_data_path', type=str, default='../data/benchmark')
    parser.add_argument('--output_path', type=str, default='../data/evaluation/cross_blink')
    parser.add_argument('--model_path', type=str, default='../models/cross_blink')
    parser.add_argument('--dataset', type=str, choices=[
        'ace2004_questions',
        'AIDA-YAGO2_testa',
        'AIDA-YAGO2_testb',
        'AIDA-YAGO2_train',
        'aquaint_questions',
        'clueweb_questions',
        'msnbc_questions',
        'wnedwiki_questions',
        'NEL',
    ], required=True)
    parser.add_argument('--score_function', type=str, choices=['add', 'mul'], default='add')
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)