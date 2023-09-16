import numpy as np
import evaluate
from transformers import EvalPrediction
from utils import postprocess_text

def load_metric(metric_name):
    if metric_name == "accuracy":
        return evaluate.load("accuracy")
    elif metric_name == "f1":
        return evaluate.load("f1")

def seq2seq_compute_metrics(tokenizer, metric):
    def compute_metrics(eval_pred: EvalPrediction):
        nonlocal tokenizer, metric
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions, labels = postprocess_text(predictions, labels)
        result = metric.compute(predictions=predictions, references=labels)
        return result
    return compute_metrics