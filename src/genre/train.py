# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, AdamW

from genre.model import GenreModel, get_model_obj, load_model
import genre.data as data
import genre.utils as utils
from genre.optimizer import get_bert_optimizer
from genre.params import ClinkParser
import genre.commons as commons

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    ranker, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        # ranker.model.eval()
        ranker.eval()
        if params["silent"]:
            iter_ = eval_dataloader
        else:
            iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0
    
        for step, batch in enumerate(iter_):
            # batch = tuple(t.to(device) for t in batch)
            context_input, labels = batch

            result = ranker.encode(context_input)

            tmp_eval_accuracy = utils.genre_accuracy(result, labels)

            eval_accuracy += tmp_eval_accuracy

            # nb_eval_examples += context_input.size(0)
            nb_eval_examples += len(context_input)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results


def get_optimizer(model, params, local_rank=0):
    # return get_bert_optimizer(
    #     [model],
    #     type_optimization='additional_layers' if (params['training_objective'] == 'category') else params["type_optimization"],
    #     learning_rate=params["learning_rate"],
    #     fp16=params.get("fp16"),
    #     local_rank=local_rank,
    # )

    optimizer = AdamW(
        model.parameters(), 
        lr=params["learning_rate"]
    )

    return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps,
    )
    if (logger):
        logger.info(" Num optimization steps = %d" % num_train_steps)
        logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(local_rank, params):
    if params["data_parallel"]:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            world_size=torch.cuda.device_count(), rank=local_rank)
        torch.cuda.set_device(local_rank)

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path) and (local_rank in [-1, 0]):
        os.makedirs(model_output_path)

    if (local_rank in [-1, 0]):
        logger = utils.setup_logger('CLINK', params["output_path"])
    else:
        logger = None

    # Init model
    ranker = GenreModel(params)
    tokenizer = ranker.tokenizer
    model = ranker
    device = ranker.device
    n_gpu = ranker.n_gpu

    ranker = ranker.to(ranker.device)

    if (params['data_parallel']):
        ranker = torch.nn.parallel.DistributedDataParallel(
            ranker, device_ids=[local_rank], 
            # output_device=local_rank,
            # find_unused_parameters=True,
        )

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_samples, train_num = data.read_dataset("train", params["data_path"], ratio=1)
    if (logger):
        logger.info("Read %d train samples." % train_num)

    processed_train_samples = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )

    train_data = data.GenreDataset(processed_train_samples)

    if (params["data_parallel"]):
        train_sampler = DistributedSampler(train_data, 
                                            num_replicas=dist.get_world_size(),
                                            rank=dist.get_rank(),
                                            shuffle=params["shuffle"])
    elif params["shuffle"]:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples, valid_num = data.read_dataset("valid", params["data_path"])
    if (logger):
        logger.info("Read %d valid samples." % valid_num)

    processed_valid_samples = data.process_mention_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )

    valid_data = data.GenreDataset(processed_valid_samples)

    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    # if local_rank in [-1, 0]:
    #     results = evaluate(
    #         ranker, valid_dataloader, params, device=device, logger=logger,
    #     )

    time_start = time.time()

    if local_rank in [-1, 0]:
        param_path = os.path.join(model_output_path, commons.config_name)
        with open(param_path, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(params))

    if (logger):
        logger.info("Start training")
        logger.info(
            "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
        )

    optimizer = get_optimizer(model, params, local_rank=local_rank)
    scheduler = get_scheduler(params, optimizer, len(train_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        if (params["data_parallel"]):
            train_sampler.set_epoch(epoch_idx)

        tr_loss = 0
        results = None

        if params["silent"] or not(local_rank in [-1, 0]):
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            # batch = tuple(t.to(device) for t in batch)
            context_input, labels = batch
            loss = ranker(
                context_input, 
                labels,
            )

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                if (logger):
                    logger.info(
                        "Step %d - epoch %d average loss: %.4f" %(
                            step,
                            epoch_idx,
                            tr_loss / (params["print_interval"] * grad_acc_steps),
                        )
                    )
                    # logger.info(semantic_scores)
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                if (logger):
                    logger.info("Evaluation on the development dataset")
                    evaluate(
                        ranker, valid_dataloader, params, device=device, logger=logger,
                    )
                    model.train()
                    logger.info("\n")

        if (logger):
            logger.info("***** Saving fine - tuned model *****")

        if local_rank in [-1, 0]:
            epoch_output_folder_path = os.path.join(
                model_output_path, "epoch_%d" %(epoch_idx)
            )
            utils.save_model(model, tokenizer, epoch_output_folder_path)

            output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
            results = evaluate(
                ranker, valid_dataloader, params, device=device, logger=logger,
            )

            ls = [best_score, results["normalized_accuracy"]]
            li = [best_epoch_idx, epoch_idx]

            best_score = ls[np.argmax(ls)]
            best_epoch_idx = li[np.argmax(ls)]
            if (logger):
                logger.info("\n")

        if (local_rank != -1):
            dist.barrier()

    execution_time = (time.time() - time_start) / 60

    if local_rank in [-1, 0]:
        utils.write_to_file(
            os.path.join(model_output_path, "training_time.txt"),
            "The training took {} minutes\n".format(execution_time),
        )
        logger.info("The training took {} minutes\n".format(execution_time))

        # save the best model in the parent_dir
        logger.info("Best performance in epoch: {}".format(best_epoch_idx))
        params["path_to_model"] = os.path.join(
            model_output_path, 
            "epoch_%d" %(best_epoch_idx),
            commons.checkpoint_name,
        )
        # TODO
        ranker = load_model(params)
        utils.save_model(ranker, tokenizer, model_output_path)

        if params["evaluate"]:
            params["path_to_model"] = model_output_path
            evaluate(ranker, valid_dataloader, params, device=device, logger=logger)


if __name__ == "__main__":
    parser = ClinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)
    print(torch.cuda.device_count())

    params = args.__dict__

    if args.data_parallel:
        print('Distributed training!!')
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '6378'
        mp.spawn(main, nprocs=torch.cuda.device_count(), args=(params, ), join=True)
    else:
        main(-1, params)