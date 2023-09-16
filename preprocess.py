import os
import json
from functools import partial

from datasets import load_dataset

from utils import preprocess_prompt, compute_similarity_sentence

id2label = {
    '0': 'negative',
    '1': 'positive'
}

def load_data_from_datahub(
        dataset_name, 
        save_data_dir,
        prompt="",
        prompt_version=None
    ):
    raw_dataset = load_dataset(dataset_name)

    for data_type in raw_dataset:
        examples = raw_dataset[data_type]
        if data_type == "train" and prompt_version in ["one-shot", "few-shot"]:
            if prompt_version == "one-shot":
                top_k_samples = 1
            else:
                top_k_samples = 2
            top_k_indices = compute_similarity_sentence(raw_dataset["train"]["text"], top_k_samples)
            raw_dataset['train'] = raw_dataset['train'].add_column("top_index", top_k_indices.values())

            corpus = raw_dataset["train"]["text"]
            labels = raw_dataset["train"]["feeling"]

            processed_train_dataset = raw_dataset["train"].map(
                partial(
                    preprocess_prompt,
                    top_k_indices=top_k_indices,
                    corpus=corpus,
                    labels=labels,
                    id2label=id2label
                ),
                batched=True,
                desc="Running prepare prompts on dataset",
            )
            in_contexts = processed_train_dataset["prompt"]
        else:
            in_contexts = ["" for i in range(len(raw_dataset["train"]["text"]))]

        sentences = []
        for index, sentence in enumerate(examples["text"]):
            if prompt_version == "zero-shot":
                sentence = prompt + sentence + ". Answer:"
            elif prompt_version in ["one-shot", "few-shot"]:
                sentence = in_contexts[index] + prompt + sentence + ". Answer:"
            sentences.append(sentence)
        
        labels = examples["feeling"]
        labels = [id2label[str(label)] for label in labels]

        save_data_file = os.path.join(save_data_dir, f"{data_type}.jsonl")
        print(f"Write into ... {save_data_file}")
        with open(save_data_file, "w") as f:
            for sentence, label in zip(sentences, labels):
                data = {
                    "sentence": sentence,
                    "label": label
                }
                print(json.dumps(data, ensure_ascii=False), file=f)