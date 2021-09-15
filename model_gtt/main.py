from clearml import Task, StorageManager, Dataset
import json
import os
import shutil
import argparse

class bucket_ops:
    StorageManager.set_cache_file_limit(5, cache_context=None)

    def list(remote_path:str):
        return StorageManager.list(remote_path, return_full_path=False)

    def upload_folder(local_path:str, remote_path:str):
        StorageManager.upload_folder(local_path, remote_path, match_wildcard=None)
        print("Uploaded {}".format(local_path))

    def download_folder(local_path:str, remote_path:str):
        StorageManager.download_folder(remote_path, local_path, match_wildcard=None, overwrite=True)
        print("Downloaded {}".format(remote_path))
    
    def get_file(remote_path:str):        
        object = StorageManager.get_local_copy(remote_path)
        return object

    def upload_file(local_path:str, remote_path:str):
        StorageManager.upload_file(local_path, remote_path, wait_for_upload=True, retries=3)

if __name__ == '__main__':
    
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

    task = Task.init(project_name='GTT', task_name='baseGTT', output_uri="s3://experiment-logging/storage/")
    clearlogger = task.get_logger()
    task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
    task.execute_remotely(queue_name="default", exit_process=True)

    print(os.getcwd())

    #Download Pretrained Models
    bucket_ops.download_folder(
        local_path="/models/bert-base-uncased", 
        remote_path="s3://experiment-logging/pretrained/bert-base-uncased", 
        )

    #Read args from config file instead, use vars() to convert namespace to dict
    dataset = Dataset.get(dataset_name="wikievents-muc4", dataset_project="datasets/wikievents", dataset_tags=["muc4-format"], only_published=True)
    dataset_folder = dataset.get_local_copy()
    print(list(os.walk(dataset_folder)))

    # if os.path.exists(dataset_folder)==False:
    os.symlink(os.path.join(dataset_folder, "data/wikievents/muc_format"), args.data_dir)

    import glob
    import logging
    from collections import OrderedDict
    from run_pl_gtt import *

    global_args = args
    logger.info(args)
    model = NERTransformer(args)
    trainer = generic_train(model, args)

    # if args.do_predict:
    #     # See https://github.com/huggingface/transformers/issues/3159
    #     # pl use this format to create a checkpoint:
    #     # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
    #     # /pytorch_lightning/callbacks/model_checkpoint.py#L169
    #     # s3object = S3Utils("blackwidow-s3", "https://ecs.dsta.ai", "AKIA8C43BC01F5E3176C", "VKYHHxqQl5GW/g3RG6c/qR65EbNrpTBBdNRtYX08", upload_multi_part=True)
    #     s3object.download_folder("models/trained_outputs/gtt/",args.output_dir)
    #     print(os.listdir(args.output_dir))
    #     # checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt'"), recursive=True)))
    #     # print(checkpoints)
    #     checkpoints = [item for item in os.listdir(args.output_dir) if "checkpoint" in item]
    #     model = NERTransformer.load_from_checkpoint(os.path.join(args.output_dir, checkpoints[-1]))
    #     # model = NERTransformer.load_from_checkpoint(checkpoints[-1])
    #     model.hparams = args
    #     if args.debug:
    #         model.hparams.debug = True
    #     trainer.test(model)
    #     preds_path = os.path.join(args.output_dir,"preds_gtt.out")
    #     s3object.upload_file(preds_path, "models/trained_outputs/gtt/preds_gtt.out")