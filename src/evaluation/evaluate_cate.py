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


def read_dataset(dataset, data_folder):
    file_name = '%s_typed.jsonl' %dataset
    json_file_path = os.path.join(data_folder, dataset, file_name)

    samples = []
    with open(json_file_path, 'rb') as fin:
        for line in fin:
            js = json.loads(line)
            samples.append(js)

    return samples, len(samples)


def to_text_cates(
    vec,
    name_map,
):
    labels = []
    for t in vec:
        tmp = []
        tmp_probs = []
        for i, x in enumerate(t):
            if (x >= 0.5):
                tmp.append((name_map[i], x))

        labels.append(tmp)

    return labels
        

def cate_accuracy(result, reference):
    m = len(result)
    tmp_acc = 0
    label_num = 0
    for _res, _ref in zip(result, reference):
        res, ref = set([x[0] for x in _res]), set([x[0] for x in _ref])
        tmp_acc += len(set(res) & set(ref))
        label_num += len(ref)
    return tmp_acc, label_num


def evaluate(
    args,
    all_records,
    ranker,
    name_map,
    _lambda=0.5,
    _threshold=0.5,
    debug=False,
):
    with torch.no_grad():
        ranker.eval()
        iter_ = tqdm(all_records, desc="Evaluation")
        device = ranker.device

        results = {}

        ctxt_eval_accuracy = 0.0
        cand_eval_accuracy = 0.0
        ctxt_nb_eval_examples = 0
        cand_nb_eval_examples = 0
        nb_eval_steps = 0
    
        nb_film = 0
        nb_musical = 0
        nb_entities = 0

        for step, record in enumerate(iter_):
            context_ids = torch.tensor(record["context"]["ids"], dtype=torch.long).to(device)
            labels = torch.tensor(record["label"], dtype=torch.float).unsqueeze(0).to(device)

            semantic_scores, category_scores, category_vec, category_ctxt, category_cand = get_model_obj(ranker).score(
                context_input=context_ids, 
            )
            
            cate_labels = record["category"]
            ctxt_category_labels = cate_labels[:, :get_model_obj(ranker).cate_num]
            cand_category_labels = cate_labels[:, get_model_obj(ranker).cate_num:]
            category_ctxt = torch.sigmoid(category_ctxt).detach().cpu().numpy()
            category_cand = torch.sigmoid(category_cand).detach().cpu().numpy()

            ctxt_category_label_names = to_text_cates(ctxt_category_labels, name_map)
            cand_category_label_names = to_text_cates(cand_category_labels, name_map)
            ctxt_category_names = to_text_cates(category_ctxt, name_map)
            cand_category_names = to_text_cates(category_cand, name_map)

            # print('Predicted context types: ' + str(ctxt_category_names))
            # print('Golden context types: ' + str(ctxt_category_label_names))
            # print('Predicted entity types: ' + str(cand_category_names))
            # print('Golden entity types: ' + str(cand_category_label_names))
            # input()

            tmp_ctxt_eval_accuracy, label_num = cate_accuracy(ctxt_category_label_names, ctxt_category_names)
            ctxt_eval_accuracy += tmp_ctxt_eval_accuracy
            ctxt_nb_eval_examples += label_num

            tmp_cand_eval_accuracy, label_num = cate_accuracy(cand_category_label_names, cand_category_names)
            cand_eval_accuracy += tmp_cand_eval_accuracy
            cand_nb_eval_examples += label_num

            nb_eval_steps += 1

        ctxt_normalized_eval_accuracy = ctxt_eval_accuracy / ctxt_nb_eval_examples
        cand_normalized_eval_accuracy = cand_eval_accuracy / cand_nb_eval_examples
        results["ctxt_normalized_accuracy"] = ctxt_normalized_eval_accuracy
        results["cand_normalized_accuracy"] = cand_normalized_eval_accuracy
        results['example_num'] = nb_eval_steps
        results['ctxt_correct_num'] = ctxt_eval_accuracy
        results['cand_correct_num'] = cand_eval_accuracy
        return results


def main(args):
    # load data
    test_samples, test_num = read_dataset(args.dataset, args.evaluation_data_path)

    # load type map
    type_map = {}
    with open(args.type_map_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            type_map[js['index']] = js['type']

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
    if (args.grid):
        result_path = os.path.join(args.output_path, '%s_grid_results.txt' %(args.dataset))
        with open(result_path, 'w', encoding='utf-8') as fout:
            for thr in np.arange(0, 1.1, 0.1):
                results = evaluate(
                    args=args,
                    all_records=test_data,
                    ranker=ranker,
                    name_map=type_map,
                    _lambda=args.lambd,
                    _threshold=thr,
                    debug=args.debug,
                )

                print('Result at threshold=%f: ' %(thr))
                print(json.dumps(results, indent=0))

                fout.write('Result at threshold=%f: \n' %(thr))
                json.dump(results, fout, indent=2)
                fout.write('\n')

    else:
        results = evaluate(
            args=args,
            all_records=test_data,
            ranker=ranker,
            name_map=type_map,
            _threshold=args.threshold,
            debug=args.debug,
        )

        print(json.dumps(results, indent=0))
        result_path = os.path.join(args.output_path, 'cate_results.txt')
        with open(result_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_data_path', type=str, default='../data/benchmark')
    parser.add_argument('--output_path', type=str, default='../data/cate_evaluation/cross_blink')
    parser.add_argument('--model_path', type=str, default='../models/cross_blink')
    parser.add_argument('--type_map_path', type=str, default='../data/vector/type_map.jsonl')
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