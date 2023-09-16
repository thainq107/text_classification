import os
import sys
import logging
from functools import partial
import numpy as np

import datasets
from datasets import DatasetDict

import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    set_seed,
)

from huggingface_hub import login, create_repo, delete_repo

from model.metric import load_metric, seq2seq_compute_metrics
from model.dataloader import load_dataset_from_path, preprocess_function
from model.model import load_model, prepare_lora_model

from utils import get_gpu_utilization

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logger = logging.getLogger(__name__)

def train(model_args, data_args, training_args, lora_args=None):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)
    # login hub
    if training_args.push_to_hub:
        login(
            token=training_args.hub_token
        )
        try:
            create_repo(training_args.hub_model_id, private=False)
        except:
            delete_repo(training_args.hub_model_id)
            create_repo(training_args.hub_model_id, private=False)

    # load dataset
    raw_dataset = load_dataset_from_path(
        data_args.save_data_dir,
        data_args.dataset_name,
        data_args.prompt_version, 
        data_args.train_file, 
        data_args.validation_file,
        data_args.test_file
    )
    raw_dataset = DatasetDict(raw_dataset)
    logger.info(f"Dataset loaded: {raw_dataset}")

    # load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = load_model(model_args)

    # prepare metric
    metric = load_metric(data_args.metric_name)
    compute_metrics = seq2seq_compute_metrics(tokenizer, metric)
    
    # prepare lora
    if lora_args is not None and lora_args.use_lora:
        model = prepare_lora_model(lora_args, model)
        model.print_trainable_parameters()

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="Dataset map pre-processing"):
        processed_dataset = raw_dataset.map(
            partial(
                preprocess_function,
                data_args=data_args,
                tokenizer=tokenizer
            ),
            batched=True,
            load_from_cache_file=False,
            remove_columns=['sentence', 'label'],
            desc="Running tokenizer on dataset",
        )

    # ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        # Start training
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(processed_dataset["train"])
        metrics["gpu_memory"] = get_gpu_utilization()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=processed_dataset["validation"])
        metrics["eval_samples"] = len(processed_dataset["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if "labels" in processed_dataset["test"].features:
            metrics = trainer.evaluate(eval_dataset=processed_dataset["test"])
            metrics["test_samples"] = len(processed_dataset["test"])
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        predictions = trainer.predict(processed_dataset["test"], metric_key_prefix="predict").predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        predictions = [str(pred).strip() for pred in predictions]
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("\n".join(predictions))
        logger.info("Predict results saved at {}".format(output_predict_file))

    # Save processor and create model card
    tokenizer.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.create_model_card()
        trainer.push_to_hub()