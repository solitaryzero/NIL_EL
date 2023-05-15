import argparse
import os
import sys


class ClinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.
    """

    def __init__(
        self, add_general_args=True, add_model_args=False, 
        description='BLINK parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_general_args,
        )

        if add_general_args:
            self.add_general_args()
        if add_model_args:
            self.add_model_args()

    def add_general_args(self, args=None):
        """
        Add common args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            default=True,
            type=bool,
            help="Whether the dataset is from zeroshot.",
        )
        parser.add_argument(
            "--mention_mask",
            default=False,
            type=bool,
            help="Whether use mask on mentions.",
        )
        parser.add_argument(
            "--single_type",
            action="store_true",
            help="Whether use single type.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")

        parser.add_argument(
            "--training_objective",
            type=str,
            choices=[
                'all',
                'semantic',
                'category'
            ],
            default='all',
        )

        # lengths
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )

        # weights
        parser.add_argument(
            "--sem_loss_weight",
            default=1.0,
            type=float,
            help="Weight of semantic loss",
        )
        parser.add_argument(
            "--cate_ctxt_loss_weight",
            default=1.0,
            type=float,
            help="Weight of context category loss",
        )
        parser.add_argument(
            "--cate_cand_loss_weight",
            default=1.0,
            type=float,
            help="Weight of candidate category loss",
        )
        parser.add_argument(
            "--sem_score_weight",
            default=1.0,
            type=float,
            help="Weight of semantic score",
        )
        parser.add_argument(
            "--cate_score_weight",
            default=1.0,
            type=float,
            help="Weight of category score",
        )

        # paths
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--path_to_config",
            default=None,
            type=str,
            required=False,
            help="The full path to the model config to load.",
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )


        # misc
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_false",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument(
            "--out_dim", type=int, default=768, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--cate_num", type=int, default=403, help="Number of category in schema.",
        )


    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--fp16", action="store_true", help="Whether to use fp16."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="Total batch size for training."
        )
        parser.add_argument("--max_grad_norm", 
            default=1.0, 
            type=float
        )
        parser.add_argument("--max_grad_value", 
            default=1.0, 
            type=float
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=10, 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=100,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", action="store_true", 
            help="Whether to shuffle train data",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for cached candidate pool (id tokenization of candidates)",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for cached candidate encoding",
        )
