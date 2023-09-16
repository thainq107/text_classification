import os

from typing import List, Optional, Union
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments
)

from transformers.trainer_utils import IntervalStrategy, HubStrategy

from preprocess import load_data_from_datahub
from train import train

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="carblacac/twitter-sentiment-analysis", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    save_data_dir: Optional[str] = field(
        default="data", metadata={"help": "A folder save the dataset."}
    )
    train_file: Optional[str] = field(
        default="train.jsonl", metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default="validation.jsonl", metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default="test.jsonl", metadata={"help": "A csv or a json file containing the test data."}
    )
    prompt_version: Optional[str] = field(
        default=None, metadata={"help": "Choose prompt version in [zero-shot]."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    metric_name: Optional[str] = field(
        default="accuracy", metadata={"help": "The metric to use for evaluation."}
    )

    def __post_init__(self):
        if self.prompt_version is not None:
            save_data_dir = os.path.join(self.save_data_dir, self.dataset_name.split("/")[-1])
            os.makedirs(save_data_dir, exist_ok=True)
            save_data_dir = os.path.join(save_data_dir, self.prompt_version)
            prompt = " Question: What is the sentiment of the following text? Choose from 'positive' or 'negative'. Text: "
        else:
            save_data_dir = os.path.join(self.save_data_dir, self.dataset_name.split("/")[-1])
            prompt = ""
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir, exist_ok=True)
            load_data_from_datahub(
                self.dataset_name, 
                save_data_dir,
                prompt,
                self.prompt_version
            )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="google/flan-t5-small", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class Seq2SeqTrainingArgumentsCustom(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="save_model", 
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})
    num_train_epochs: float = field(
        default=10.0, metadata={"help": "Total number of training epochs to perform."}
    )
    logging_dir: Optional[str] = field(
        default="logs", metadata={"help": "Tensorboard log dir."}
    )
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default="epoch", metadata={"help": "The checkpoint save strategy to use."},
    )
    # save_steps: float = field(
    #     default=500,
    #     metadata={
    #         "help": (
    #             "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
    #             "If smaller than 1, will be interpreted as ratio of total training steps."
    #         )
    #     },
    # )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": ("If a value is passed, will limit the total amount of checkpoints.")},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True, 
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    report_to: Optional[List[str]] = field(
        default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    metric_for_best_model: Optional[str] = field(
        default="accuracy", metadata={"help": "The metric to use to compare two different models."}
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_strategy: Union[HubStrategy, str] = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )

@dataclass
class LoraArguments:
    """
    Arguments pertaining to which lora we are going to fine-tune from.
    """

    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Use LORA or not"}
    )
    lora_r: Optional[float] = field(
        default=8, metadata={"help": "the rank of the update matrices, expressed in int. "
                                "Lower rank results in smaller update matrices with fewer trainable parameters."}
    )
    lora_alpha: Optional[float] = field(
        default=32, metadata={"help": "LoRA scaling factor."}
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q", "v","k"],
        metadata={"help": "The modules (for example, attention blocks) to apply the LoRA update matrices."},
    )
    lora_bias: Optional[str] = field(
        default="none",
        metadata={"help": "Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'."},
    )
    
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArgumentsCustom, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.prompt_version is not None:
        save_folder_path = model_args.model_name_or_path.split("/")[-1] + "-" + data_args.dataset_name.split("/")[-1] + "-" + data_args.prompt_version
    else:
        save_folder_path = model_args.model_name_or_path.split("/")[-1] + "-" + data_args.dataset_name.split("/")[-1]

    training_args.hub_model_id = save_folder_path
    if training_args.fp16:
        training_args.hub_model_id = training_args.hub_model_id + "-fp16"
        save_folder_path = save_folder_path + "-fp16"
    elif training_args.bf16:
        training_args.hub_model_id = training_args.hub_model_id + "-bf16"
        save_folder_path = save_folder_path + "-bf16"
    
    if lora_args.use_lora:
        training_args.hub_model_id = training_args.hub_model_id + "-lora"
        save_folder_path = save_folder_path + "-lora"

    training_args.output_dir = os.path.join(training_args.output_dir, save_folder_path)
    training_args.logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
    train(model_args, data_args, training_args, lora_args)

if __name__ == "__main__":
    main()