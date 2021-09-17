from clearml import Task, StorageManager, Dataset
import json
import os
import shutil
import argparse
from pathlib import Path

task = Task.init(project_name='GTT-test', task_name='baseGTT-2', output_uri="s3://experiment-logging/storage/")
task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)

parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)

parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--n_tpu_cores", type=int, default=0)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)

parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list",
    )
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: ",
)
parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
)

parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument(
        "--max_seq_length_src",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization for src. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

parser.add_argument(
    "--max_seq_length_tgt",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization for tgt. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)

parser.add_argument(
    "--labels",
    default="",
    type=str,
    help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
)

parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
)

parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)

parser.add_argument("--debug", action="store_true", help="if in debug mode")

parser.add_argument("--thresh", default=1, type=float, help="thresh for predicting [SEP]",)

# add_generic_args(parser, os.getcwd())
# parser = NERTransformer.add_model_specific_args(parser, os.getcwd())
args = parser.parse_args()
task.connect(args)
task.execute_remotely(queue_name="128RAMv100", exit_process=True)
