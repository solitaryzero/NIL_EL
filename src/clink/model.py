# coding: utf-8

import clink.commons as commons

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


def load_model(params):
    return ClinkModel(params)


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


class ClinkModel(nn.Module):
    def __init__(self, params):
        super(ClinkModel, self).__init__()
        # init device
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # init structure
        ctxt_bert = BertModel.from_pretrained(params['bert_model'])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        self.ctxt_labeler = nn.Linear(self.context_encoder.out_dim, params['cate_num'])
        self.cand_labeler = nn.Linear(self.cand_encoder.out_dim, params['cate_num'])

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
        mask = (token_idx != null_idx)
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask


    def encode(
        self,
        context_input,
        cand_input,
        context_mask=None,
        candidate_mask=None,
    ):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )
        token_idx_cands, segment_idx_cands, mask_cands = self.to_bert_input(
            cand_input, self.NULL_IDX
        )

        embedding_ctxt = None
        category_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, context_mask
            )
            category_ctxt = self.ctxt_labeler(embedding_ctxt)

        embedding_cands = None
        category_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands, candidate_mask
            )
            category_cands = self.cand_labeler(embedding_cands)
        
        return embedding_ctxt, embedding_cands, category_ctxt, category_cands


    def encode_context(self, context_input, context_mask=None):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )

        embedding_ctxt = None
        category_ctxt = None
        embedding_ctxt = self.context_encoder(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, context_mask
        )
        category_ctxt = self.ctxt_labeler(embedding_ctxt)

        return embedding_ctxt, category_ctxt

    
    def encode_candidate(self, cand_input, candidate_mask=None):
        token_idx_cands, segment_idx_cands, mask_cands = self.to_bert_input(
            cand_input, self.NULL_IDX
        )

        embedding_cands = None
        category_cands = None
        embedding_cands = self.cand_encoder(
            token_idx_cands, segment_idx_cands, mask_cands, candidate_mask
        )
        category_cands = self.cand_labeler(embedding_cands)

        return embedding_cands, category_cands


    def score(
        self,
        context_input,
        cand_input,
        context_mask,
        candidate_mask,
    ):
        embedding_ctxt, embedding_cands, category_ctxt, category_cands = self.encode(
            context_input=context_input,
            cand_input=cand_input,
            context_mask=context_mask,
            candidate_mask=candidate_mask,
        )

        semantic_scores = embedding_ctxt.mm(embedding_cands.t())
        semantic_scores = torch.sigmoid(semantic_scores)
        
        category_ctxt = torch.sigmoid(category_ctxt)
        category_cands = torch.sigmoid(category_cands)
        category_scores = category_ctxt.mm(category_cands.t())

        semantic_scores = semantic_scores.cpu().detach().numpy()
        category_scores = category_scores.cpu().detach().numpy()

        return semantic_scores, category_scores


    def score_candidates(
        self,
        context_input,
        cand_inputs,
        context_mask=None,
    ):
        embedding_ctxt, category_ctxt = self.encode_context(context_input)
        assert cand_inputs.shape[0] == 1
        cand_inputs = cand_inputs.squeeze(0)
        embedding_cands, category_cands = self.encode_candidate(cand_inputs)

        semantic_scores = embedding_ctxt.mm(embedding_cands.t())
        semantic_scores = torch.sigmoid(semantic_scores)
        category_scores = F.cosine_similarity(category_ctxt, category_cands, dim=1)
        category_scores = category_scores * 0.5 + 0.5

        return semantic_scores, category_scores


    def score_hard_candidates(
        self,
        context_input,
        cand_input,
    ):
        embedding_ctxt, category_ctxt = self.encode_context(context_input)
        embedding_cand, category_cand = self.encode_candidate(cand_input)
        semantic_scores = (embedding_ctxt * embedding_cand).sum(dim=1)
        
        if (self.params['single_type']):
            category_scores = F.cosine_similarity(category_ctxt, category_cand, dim=1)
        else:
            _category_ctxt = torch.sigmoid(category_ctxt)
            _category_cand = torch.sigmoid(category_cand)
            category_scores = F.cosine_similarity(_category_ctxt, _category_cand, dim=1)
            category_scores = category_scores * 0.5 + 0.5

        return semantic_scores, category_scores, category_ctxt, category_cand


    def forward(
        self, 
        context_input, 
        cand_input, 
        ctxt_category_input,
        cand_category_input,
        context_mask=None,
        candidate_mask=None,
        label=None,
        lambd=None,
        hard_negative=False,
    ):
        if hard_negative:
            assert (label is not None)
            semantic_scores, category_scores, category_ctxt_pred, category_cand_pred = self.score_hard_candidates(
                context_input=context_input,
                cand_input=cand_input,
            )
            if (lambd != None):
                final_score = lambd*semantic_scores+(1-lambd)*category_scores
            else:
                final_score = semantic_scores*category_scores
            sem_loss = F.binary_cross_entropy_with_logits(semantic_scores, label.float())

            if (self.params['single_type']):
                ctxt_category_input = torch.argmax(ctxt_category_input, dim=1)
                cand_category_input = torch.argmax(cand_category_input, dim=1)
                category_ctxt_loss = F.cross_entropy(category_ctxt_pred, ctxt_category_input, reduction="mean")
                category_cand_loss = F.cross_entropy(category_cand_pred, cand_category_input, reduction="mean")
            else:
                # category_ctxt_loss = F.multilabel_soft_margin_loss(category_ctxt_pred, ctxt_category_input, reduction="mean")
                # category_cand_loss = F.multilabel_soft_margin_loss(category_cand_pred, cand_category_input, reduction="mean")
                fl = FocalLoss(gamma=2)
                category_ctxt_loss = fl(category_ctxt_pred, ctxt_category_input, reduction="mean")
                category_cand_loss = fl(category_cand_pred, cand_category_input, reduction="mean")
            return sem_loss, category_ctxt_loss, category_cand_loss, final_score

        else:
            embedding_ctxt, embedding_cands, category_ctxt, category_cands = self.encode(
                context_input=context_input,
                cand_input=cand_input,
                context_mask=context_mask,
                candidate_mask=candidate_mask,
            )

            semantic_scores = embedding_ctxt.mm(embedding_cands.t())
            batch_size = semantic_scores.size(0)
            target = torch.LongTensor(torch.arange(batch_size))
            target = target.to(self.device)
            semantic_loss = F.cross_entropy(semantic_scores, target, reduction="mean")

            if (self.params['single_type']):
                ctxt_category_input = torch.argmax(ctxt_category_input, dim=1)
                cand_category_input = torch.argmax(cand_category_input, dim=1)
                category_ctxt_loss = F.cross_entropy(category_ctxt_pred, ctxt_category_input, reduction="mean")
                category_cand_loss = F.cross_entropy(category_cand_pred, cand_category_input, reduction="mean")
            else:
                # category_ctxt_loss = F.multilabel_soft_margin_loss(category_ctxt, ctxt_category_input, reduction="mean")
                # category_cand_loss = F.multilabel_soft_margin_loss(category_cands, cand_category_input, reduction="mean")

                fl = FocalLoss(gamma=2)
                category_ctxt_loss = fl(category_ctxt, ctxt_category_input, reduction="sum")
                category_cand_loss = fl(category_cands, cand_category_input, reduction="sum")

            return semantic_loss, category_ctxt_loss, category_cand_loss, semantic_scores