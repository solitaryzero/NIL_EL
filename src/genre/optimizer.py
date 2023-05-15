import torch
import os
import numpy as np

from transformers import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from transformers import BertTokenizer
from torch import nn

from transformers import AdamW


patterns_optimizer = {
    'additional_layers': ['additional', 'labeler'],
    'top_layer': ['additional', 'bert_model.encoder.layer.11.', 'labeler'],
    'top4_layers': [
        'additional',
        'bert_model.encoder.layer.11.',
        'encoder.layer.10.',
        'encoder.layer.9.',
        'encoder.layer.8',
        'labeler',
    ],
    'all_encoder_layers': ['additional', 'bert_model.encoder.layer', 'labeler'],
    'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings', 'labeler'],
}

def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False, local_rank=0):
    """ Optimizes the network with AdamWithDecay
    """
    if type_optimization not in patterns_optimizer:
        if (local_rank == 0):
            print(
                'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
            )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    if (local_rank == 0):
        print('The following parameters will be optimized WITH decay:')
        print(ellipse(parameters_with_decay_names, 5, ' , '))
        print('The following parameters will be optimized WITHOUT decay:')
        print(ellipse(parameters_without_decay_names, 5, ' , '))

    # if (local_rank == 0):
    #     print('The following parameters will be optimized WITH decay:')
    #     print(parameters_with_decay_names)
    #     print('The following parameters will be optimized WITHOUT decay:')
    #     print(parameters_without_decay_names)

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate, 
        correct_bias=False
    )

    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)
