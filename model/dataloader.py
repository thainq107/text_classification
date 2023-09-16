import os

from datasets import load_dataset

def load_dataset_from_path(save_data_dir, dataset_name, prompt_version, train_file, validation_file, test_file):
    if prompt_version is not None:
        save_data_dir = os.path.join(save_data_dir, dataset_name.split("/")[-1], prompt_version)
    else:
        save_data_dir = os.path.join(save_data_dir, dataset_name.split("/")[-1])

    print(f"Load data from: {save_data_dir}")
    train_file_path = os.path.join(save_data_dir, train_file)
    train_dataset = load_dataset('json', data_files=train_file_path)

    validation_file_path = os.path.join(save_data_dir, validation_file)
    validation_dataset = load_dataset('json', data_files=validation_file_path)

    test_file_path = os.path.join(save_data_dir, test_file)
    test_dataset = load_dataset('json', data_files=test_file_path)

    return {
        'train': train_dataset['train'],
        'validation': validation_dataset['train'],
        'test': test_dataset['train']
    }

def preprocess_function(examples, data_args, tokenizer):
    # Tokenize the texts
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    model_inputs = tokenizer(
        examples["sentence"], 
        padding="max_length", 
        max_length=max_seq_length, 
        truncation=True
    )
    labels = tokenizer(
        examples["label"],
        padding=True
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    return model_inputs