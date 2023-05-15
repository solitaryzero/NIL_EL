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

from clink.model import ClinkModel, get_model_obj, load_model
import clink.data as data
import clink.utils as utils
from clink.params import ClinkParser
import clink.commons as commons


def evaluate(
    args,
    all_records,
    ranker,
    _lambda=0.5,
    _threshold=0.5,
    debug=False,
):
    with torch.no_grad():
        ranker.eval()
        iter_ = tqdm(all_records, desc="Evaluation")
        device = ranker.device

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        nil_eval_accuracy = 0.0
        nb_nil_examples = 0

        normal_eval_accuracy = 0.0
        nb_normal_examples = 0
    
        for step, record in enumerate(iter_):
            context_ids = torch.tensor(record["context"]["ids"], dtype=torch.long).unsqueeze(0).to(device)
            context_mask = torch.tensor(record["context"]["mask"], dtype=torch.long).unsqueeze(0).to(device)
            cand_ids = torch.tensor(record["cand"]["ids"], dtype=torch.long).unsqueeze(0).to(device)
            labels = torch.tensor(record["label"], dtype=torch.float).unsqueeze(0).to(device)

            semantic_scores, category_scores = get_model_obj(ranker).score_candidates(
                context_input=context_ids, 
                cand_inputs=cand_ids,
            )

            if (args.score_function == 'add'):
                logits = _lambda*semantic_scores+(1-_lambda)*category_scores
            else:
                logits = semantic_scores*category_scores

            # print(logits)
            # print(semantic_scores)
            # print(category_scores)
            # input()

            logits = logits.squeeze(0).detach().cpu().numpy()
            labels = labels.squeeze(0).detach().cpu().numpy()

            tmp_eval_accuracy, is_nil = utils.nil_accuracy(logits, labels, threshold=_threshold)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += 1
            nb_eval_steps += 1

            if (is_nil):
                nil_eval_accuracy += tmp_eval_accuracy
                nb_nil_examples += 1
            else:
                normal_eval_accuracy += tmp_eval_accuracy
                nb_normal_examples += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        results["normalized_accuracy"] = normalized_eval_accuracy
        results['example_num'] = nb_eval_examples
        results['correct_num'] = eval_accuracy
        if (nb_nil_examples == 0):
            results['nil_accuracy'] = 0
        else:
            results['nil_accuracy'] = nil_eval_accuracy / nb_nil_examples
        results['nil_num'] = nb_nil_examples
        results['normal_accuracy'] = normal_eval_accuracy / nb_normal_examples
        results['normal_num'] = nb_normal_examples
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

    ranker = ClinkModel(params)
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
    if (args.grid):
        result_path = os.path.join(args.output_path, '%s_grid_results.txt' %(args.dataset))
        with open(result_path, 'w', encoding='utf-8') as fout:
            for lam in np.arange(0, 1.1, 0.1):
                results = evaluate(
                    args=args,
                    all_records=test_data,
                    ranker=ranker,
                    _lambda=lam,
                    _threshold=args.threshold,
                    debug=args.debug,
                )

                print('Result at lambda=%f: ' %(lam))
                print(json.dumps(results, indent=0))

                fout.write('Result at lambda=%f: \n' %(lam))
                json.dump(results, fout, indent=2)
                fout.write('\n')

    else:
        results = evaluate(
            args=args,
            all_records=test_data,
            ranker=ranker,
            _lambda=args.lambd,
            _threshold=args.threshold,
            debug=args.debug,
        )

        print(json.dumps(results, indent=0))
        result_path = os.path.join(args.output_path, '%s_thr%f_results.txt' %(args.dataset, args.threshold))
        with open(result_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_data_path', type=str, default='../data/benchmark')
    parser.add_argument('--output_path', type=str, default='../data/evaluation/raw_blink')
    parser.add_argument('--model_path', type=str, default='../models/raw_blink')
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
    parser.add_argument('--grid', action='store_true')

    args = parser.parse_args()
    main(args)