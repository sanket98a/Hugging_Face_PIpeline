import torch
from datasets import load_dataset
from peft import LoraConfig,get_peft_model,prepare_model_for_int8_training
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments
from trl import SFTTrainer

def train():
    # load the dataset from hugging face
    train_dataset=load_dataset("tatsu-lab/alpaca",split='train')
    print('Datset Loaded Succesfully...')

    tokenizer= AutoTokenizer.from_pretrained('salesforce/xgen-7b-8k-base',trust_remote_code=True)
    tokenizer.pad_token=tokenizer.eos_token
    print('Tokenizer Loaded Succesfully...')

    model=AutoModelForCausalLM.from_pretrained('salesforce/xgen-7b-8k-base',load_in_4bit=True,torch_dtype=torch.float16)
    model.reseize_token_embeddings(len(tokenizer))
    print('Model Loaded Succesfully...')
    model=prepare_model_for_int8_training(model)
    print('Prepare Model For Int8 Training Loaded Succesfully...')
    

    peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CASUAL_LM")
    model=get_peft_model(model,peft_config)
    print('Peft Model Loaded Succesfully...')

    training_args=TrainingArguments(
        output_dir="xgen-7b-tuned-alpaca",
        per_device_train_batch_size=4,
        optim='adamw_torch',
        logging_steps=100,
        learining_rate=2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=True
    )
    print('TrainingArguments Loaded Succesfully...')
    trainer=SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config
    )
    print('SFTTrainer Loaded Succesfully...')

    print('Model Training Start...')
    trainer.train()
    print('Model Trained Succesfully...')
    trainer.push_to_hub()
    print("Model Trained and push to hub")
if __name__=="__main__":
    train()
