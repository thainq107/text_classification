# Text Classification using Pre-trained Language Models: Fine-Tuning BERTs and Prompting LLMs (Flan-T5)

## Dependencies
- Python 3.10
- [PyTorch](https://github.com/pytorch/pytorch) 2.0 +
  ```
  pip install -r requirements.txt
  ```
  ## Dataset
  [carblacac/twitter-sentiment-analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)

  ## Fine-Tuning BERTs
  ### Training
  ```
    python run_fine_tuning_bert.py \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --model_name_or_path bert-base-uncased \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```

  ### Predict
  Load model from huggingface repository
  ```
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "thainq107/flan-t5-small-twitter-sentiment-analysis"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer("I hate you:", return_tensors="pt")
    outputs = model.generate(**inputs)
    tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # ['negative']
  ```
  ## Fine-Tuning Large Language Models (Flan-T5) with Prompting Techniques
  Prompting Versions: Zero-Shot, One-Shot, Few-Shot (num examples = 2)
  Prompting Selection: Cosine similarity (sentence-transformers)

  ### Training
  ```
    python run_fine_tuning_prompting \
        --dataset_name carblacac/twitter-sentiment-analysis \
        --prompt_version zero-shot \
        --model_name_or_path google/flan-t5-small \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```

  ### Predict
  Load model from huggingface repository
  zero-shot: thainq107/flan-t5-small-twitter-sentiment-analysis-zero-shot
  ```
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "thainq107/flan-t5-small-twitter-sentiment-analysis-zero-shot"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer("I hate you:", return_tensors="pt")
    outputs = model.generate(**inputs)
    tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # ['negative']
  ```
