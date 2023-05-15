# coding: utf-8

import cross.commons as commons

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


def load_model(params):
    return CrossModel(params)


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):
        loss = -(target * torch.pow(torch.sigmoid(-input), self.gamma) * F.logsigmoid(input) + 
                (1 - target) * torch.pow(torch.sigmoid(input), self.gamma) * F.logsigmoid(-input))
        class_dim = input.dim() - 1
        C = input.size(class_dim)
        loss = loss.sum(dim=class_dim) / C
        if reduction == "none":
            ret = loss
        elif reduction == "mean":
            ret = loss.mean()
        elif reduction == "sum":
            ret = loss.sum()
        else:
            ret = input
            raise ValueError(reduction + " is not valid")
        return ret


class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.config.hidden_size

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
            self.out_dim = output_dim
        else:
            self.additional_linear = None
            self.out_dim = bert_output_dim


    def forward(self, token_ids, segment_ids, attention_mask, token_mask=None):
        outputs = self.bert_model(
            input_ids=token_ids, 
            token_type_ids=segment_ids, 
            attention_mask=attention_mask
        )

        add_mask = token_mask is not None

        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif add_mask:
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs.pooler_output

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(embeddings)
        elif add_mask:
            assert token_mask is not None
            embeddings = embeddings * token_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1)
            embeddings = embeddings / token_mask.sum(-1, keepdim=True)
            result = embeddings
        else:
            result = embeddings

        return result


class CrossModel(nn.Module):
    def __init__(self, params):
        super(CrossModel, self).__init__()
        # init device
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # init structure
        cross_bert = BertModel.from_pretrained(params['bert_model'])
        self.encoder = BertEncoder(
            cross_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )

        self.score_layer = nn.Linear(self.encoder.out_dim, 1)

        self.labeler = nn.Sequential(
            nn.Linear(self.encoder.out_dim, params['cate_num']*2),
        )
        self.cate_num = params['cate_num']

        self.config = cross_bert.config
        
        # load params
        if (params['path_to_model'] is not None):
            self.load_model(params['path_to_model'])

        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def to_bert_input(self, token_idx, null_idx):
        """ token_idx is a 2D tensor int.
            return token_idx, segment_idx and mask
        """
        segment_idx = token_idx * 0
        mask = token_idx != null_idx
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask


    def encode(
        self,
        context_input,
    ):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )

        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

            category_vec = self.labeler(embedding_ctxt)
        
        return embedding_ctxt, category_vec


    def score(
        self,
        context_input,
    ):
        embedding_ctxt, category_vec = self.encode(
            context_input=context_input,
        )
        semantic_scores = self.score_layer(embedding_ctxt).squeeze(dim=1)

        category_ctxt, category_cand = category_vec[:, :self.cate_num], category_vec[:, self.cate_num:]
        if (self.params['single_type']):
            category_scores = F.cosine_similarity(category_ctxt, category_cand, dim=1)
            category_scores = category_scores * 0.5 + 0.5
        else:
            category_scores = F.cosine_similarity(category_ctxt, category_cand, dim=1)
            category_scores = category_scores * 0.5 + 0.5

        return semantic_scores, category_scores, category_vec, category_ctxt, category_cand


    def forward(
        self, 
        context_input, 
        category_input,
        label,
    ):
        cate_mask = torch.any((category_input != 0), dim=1, keepdim=False)

        semantic_scores, category_scores, category_vec, category_ctxt, category_cand = self.score(
            context_input=context_input,
        )
    
        sem_loss = F.binary_cross_entropy_with_logits(semantic_scores, label.float())
        if (self.params['single_type']):
            ctxt_category_input = category_input[:, :self.cate_num]
            cand_category_input = category_input[:, self.cate_num:]
            ctxt_category_input = torch.argmax(ctxt_category_input, dim=1)
            cand_category_input = torch.argmax(cand_category_input, dim=1)
            ctxt_cate_loss = F.cross_entropy(category_ctxt, ctxt_category_input, reduction="mean")
            cand_cate_loss = F.cross_entropy(category_cand, cand_category_input, reduction="mean")
            cate_loss = ctxt_cate_loss + cand_cate_loss
        else:
            fl = FocalLoss(gamma=2)
            cate_loss = fl(category_vec, category_input, reduction="mean")

        return sem_loss, cate_loss, semantic_scores, category_scores