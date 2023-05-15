# coding: utf-8

import cross.commons as commons

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


def load_model(params):
    return GenreModel(params)


class GenreModel(nn.Module):
    def __init__(self, params):
        super(GenreModel, self).__init__()
        # init device
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # init structure
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

        self.config = self.bart.config
        
        # load params
        if (params['path_to_model'] is not None):
            self.load_model(params['path_to_model'])

        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def encode(
        self,
        context_input,
    ):
        tokenize_result = self.tokenizer.batch_encode_plus(
            context_input, 
            padding=True, 
            max_length=256, 
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        summary_ids = self.bart.generate(
            input_ids=tokenize_result['input_ids'], 
            attention_mask=tokenize_result['attention_mask'], 
            num_beams=5,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


    def forward(
        self, 
        context_input, 
        label,
    ):
        tokenize_result = self.tokenizer.batch_encode_plus(
            context_input, 
            padding=True, 
            max_length=256, 
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        label_tokenize_result = self.tokenizer.batch_encode_plus(
            label, 
            padding=True, 
            max_length=256, 
            truncation=True,
            return_tensors="pt",
        ).to(self.device)['input_ids']
        label_tokenize_result[label_tokenize_result[:, :] == self.config.pad_token_id] = -100

        result = self.bart(
            input_ids=tokenize_result['input_ids'], 
            attention_mask=tokenize_result['attention_mask'], 
            labels=label_tokenize_result,
        )
        return result.loss