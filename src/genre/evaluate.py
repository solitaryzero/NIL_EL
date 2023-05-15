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

from genre.model import GenreModel, get_model_obj, load_model
import genre.data as data
import genre.utils as utils
from genre.params import ClinkParser
import genre.commons as commons


def evaluate(
    args,
    all_records,
    ranker,
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
            # context_ids, cand_ids, cate_vecs, context_mask, candidate_mask, labels = batch
            context_ids = [record["context"]]
            labels = [record["label"]]

            result = get_model_obj(ranker).encode(
                context_input=context_ids, 
            )

            count, nil_count, nil_num, normal_count, normal_num = utils.nil_accuracy(result, labels)

            eval_accuracy += count
            nb_eval_examples += 1
            nb_eval_steps += 1

            nil_eval_accuracy += nil_count
            nb_nil_examples += nil_num
            normal_eval_accuracy += normal_count
            nb_normal_examples += normal_num

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        # logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        results['example_num'] = nb_eval_examples
        results['correct_num'] = eval_accuracy
        results['nil_accuracy'] = nil_eval_accuracy / nb_nil_examples
        results['nil_num'] = nb_nil_examples
        results['normal_accuracy'] = normal_eval_accuracy / nb_normal_examples
        results['normal_num'] = nb_normal_examples
        return results


def main(args):
    # load data
    # evaluation_data_path = os.path.join(args.evaluation_data_path, args.dataset, 'tensor.pkl')
    test_samples, test_num = data.read_dataset("valid", args.evaluation_data_path)

    # load model
    path_to_config = os.path.join(args.model_path, commons.config_name)
    path_to_model = os.path.join(args.model_path, commons.checkpoint_name)

    with open(path_to_config, 'r', encoding='utf-8') as fin:
        params = json.load(fin)
        params["output_path"] = args.output_path
        params['path_to_model'] = path_to_model

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ranker = GenreModel(params)
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
    test_data = data.process_mention_data(
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
        ranker=ranker,
        debug=args.debug,
    )

    print(json.dumps(results, indent=0))
    result_path = os.path.join(args.output_path, '%s_results.txt' %(args.dataset))
    with open(result_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_data_path', type=str, default='/data/zfw/NEL/data/benchmark')
    parser.add_argument('--output_path', type=str, default='/data/zfw/NEL/data/evaluation/cross_blink')
    parser.add_argument('--model_path', type=str, default='/data/zfw/NEL/models/cross_blink')
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
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)