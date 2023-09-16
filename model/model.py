from transformers import AutoConfig, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType

def load_model(model_args):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task="text-generation"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    return model

def prepare_lora_model(lora_args, model):
    # creating model
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        task_type=TaskType.SEQ_2_SEQ_LM, 
        lora_alpha=lora_args.lora_alpha, 
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules,
        bias=lora_args.lora_bias
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model