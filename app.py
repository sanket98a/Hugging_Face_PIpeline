# import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from src.model import LLMModel
import yaml

with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# TEST CASE 1 :: Self Quantization using BitsAndBytes
model=LLMModel(model_id=config['MODEL_ID'])
q_config=model.quantization(load_in_4bit=True)
model.model_initialize(quantization_config=q_config,quantization=True)

