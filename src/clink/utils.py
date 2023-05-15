import json
import logging
import os
import sys
import torch
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import clink.commons as commons


def setup_logger(name, save_dir, filename="log.txt", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def category_scores(out, labels, threshold=0.5):
    out = out.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    scores = out*labels+(1-out)*(1-labels)
    return np.sum(scores >= threshold), scores >= threshold


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels


def binary_accuracy(out, labels):
    outputs = out
    outputs = np.greater_equal(outputs, 0.5, dtype=np.float32)
    return np.sum(outputs == labels), outputs == labels


def nil_accuracy(out, labels, threshold=0.5):
    is_nil = np.all(labels == 0)
    p = -1
    max_prob = 0.0
    for i, x in enumerate(out):
        if (x >= threshold) and (x > max_prob):
            max_prob = x
            p = i

    if is_nil:
        if (p == -1):
            return 1, True
        else:
            return 0, True
    else:
        if (p == -1) or (labels[p] != 1):
            return 0, False
        else:
            return 1, False


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def write_to_file(path, content):
    with open(path, 'w', encoding='utf-8') as fout:
        fout.write(content)


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, commons.checkpoint_name)
    torch.save(model_to_save.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_dir)