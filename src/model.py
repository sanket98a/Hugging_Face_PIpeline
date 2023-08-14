"""
In this `model.py` file created the Open Source Hugging Face LLM Model Loading Pipeline.
"""
# Import the required Packages
# import torch
## from peft import PeftModel, PeftConfig
from transformers import (AutoModelForCausalLM,
                           AutoTokenizer, 
                           BitsAndBytesConfig, 
                           pipeline,
                           LlamaForCausalLM)
import logging
from huggingface_hub import login
# from logger import logging
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline, LlamaCpp
from auto_gptq import AutoGPTQForCausalLM
import torch

access_token_read ="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"
login(token = access_token_read)
logging.info("Login Successfully.")
class LLMModel:
    """Hugging Face Open Source Model Loading..
    """
    def __init__(self,model_id:str,model_basename:str=None,device_type:str='cpu') -> None:
        self.model_id=model_id
        self.model_basename=model_basename
        self.device_type=device_type

    def quantization(self,
                     load_in_4bit:bool=False,
                     load_in_8bit:bool=False,
                     bnb_4bit_quant_type:str({'fp4','nf4'})='fp4',
                     bnb_4bit_compute_dtype: None = None,
                     bnb_4bit_use_double_quant=False,
                     quantization_type:str="BitsAndBytes")->object:
        """use this method to perform quantization of LLM models.
            you can quantize either 4_bit or 8_bit using BitsAndBytesConfig.

            Args:
                quantization_type(str):Default `BitsAndBytes`,but you can select any type quantization from this list ['BitsAndBytes','GPTQ]
        """
        try:
            if quantization_type=='BitsAndBytes':
                print("BitsAndBytes uantization ")
                logging.info(f"Loaded BitsAndBytes quantization config..")
                bnb_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                                load_in_8bit=load_in_8bit,
                                                bnb_4bit_quant_type=bnb_4bit_quant_type,
                                                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                                                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant)
                logging.info(f"Loaded quantization config..")
                return bnb_config
            else:
                print("GPTQ uantization ")
                logging.info(f"Loaded GPTQ quantization config..")

        except:
            pass
    
        
        return bnb_config
    
    def model_initialize(self,quantization_config:object=None,
                         quantization=False,
                         quantization_type:str="BitsAndBytes")->object:
        # try:
            logging.info(f"Loading Model: {self.model_id}, on: {self.device_type}")
            logging.info("This action can take a few minutes!")

            # Self Quantization using BitsAndBytes and GPTQ
            if quantization==True and self.device_type.lower()=='cuda':
                if quantization_type=='BitsAndBytes':
                    print("Quantization Model Loading Start..")
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, 
                                                                quantization_config = quantization_config,
                                                                device_map={"":0})
                    print("Quantization Model Load Succesfully..")
                    logging.info(f"Quantization Model Loaded :{self.model_id}")
                    return model
                else:
                    pass
            else:
                # In this else statement either you load quantized model from hugging face
                # otherwise loading full original models
                if self.model_basename is not None:
                    if ".ggml" in self.model_basename:
                        logging.info("Using Llamacpp for GGML quantized models")
                        model_path = hf_hub_download(repo_id=self.model_id, filename=self.model_basename)
                        # set path of alderly saved model
                        # model_path=r"D:\1. Ai Practices\11. Open_source_models_locally_deploy\1. Local Model\Local_llm\Local_Model\models--TheBloke--Llama-2-7B-Chat-GGML\snapshots\b616819cd4777514e3a2d9b8be69824aca8f5daf\llama-2-7b-chat.ggmlv3.q4_0.bin"
                        
                        print('*'*100)
                        print(model_path)
                        print('*'*100)
                        max_ctx_size = 2048
                        kwargs = {
                            "model_path": model_path,
                            "n_ctx": max_ctx_size,
                            "max_tokens": max_ctx_size,
                            }
                        if self.device_type.lower() == "mps":
                            kwargs["n_gpu_layers"] = 1000
                        if self.device_type.lower() == "cuda":
                            kwargs["n_gpu_layers"] = 1000
                            kwargs["n_batch"] = max_ctx_size
                        print("GGML Model Loaded Succesfully.")
                        return LlamaCpp(**kwargs)
                    
                    else:
                        # The code supports all huggingface models that ends with GPTQ and have some variation
                        # of .no-act.order or .safetensors in their HF repo.
                        logging.info("Using AutoGPTQForCausalLM for quantized models")
                        if ".safetensors" in self.model_basename :
                            # Remove the ".safetensors" ending if present
                            self.model_basename = self.model_basename .replace(".safetensors", "")
                        print("Model Name ::",self.model_basename )
                        model = AutoGPTQForCausalLM.from_quantized(
                            self.model_id,
                            model_basename=self.model_basename ,
                            use_safetensors=True,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            # device="cuda:0",
                            use_triton=False,
                            quantize_config=None)
                        return model
                    
                elif (self.device_type.lower() == "cuda"):  
                    # The code supports all huggingface models that ends with -HF or which have a .bin
                        # file in their HF repo.
                        logging.info("Using AutoModelForCausalLM for full models")
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
                        )
                        model.tie_weights()
                        print("Model Load Succesfully..")
                        logging.info(f"Model Loaded :{self.model_id}")
                        return model
                else:
                    print("Loading Full model on Cpu")
                    print("Using Llamamodel")
                    model = LlamaForCausalLM.from_pretrained(self.model_id)
                    return model
        # except:
        #     pass
        
    def tokenizer_initialize(self):
        # try:
            if ".ggml" in self.model_basename:
                logging.info(f"Tokenizer Not Required:{self.model_id}")
                
            else:
                print("Tokenizer Loading Start..")
                logging.info(f"Tokenizer Loading start..")
                tokenizer = AutoTokenizer.from_pretrained(self.model_id,use_fast=True)
                print("Tokenizer Loaded Succesfully..")
                logging.info("Tokenizer loaded.")
                return tokenizer
        # except:
        #     pass
        
    def model_pipeline(self,
                       task:str="text-generation",
                    load_in_4bit:bool=False,
                     load_in_8bit:bool=False,
                     bnb_4bit_quant_type:str({'fp4','nf4'})='fp4',
                     bnb_4bit_compute_dtype: None = None,
                     bnb_4bit_use_double_quant=False,
                     max_new_tokens:int=512,
                     temperature:float=0.7,
                     quantization:bool=False,
                     torch_dtype:any=None):
        try:
            if ".ggml" in self.model_basename:
                logging.info(f"Tokenizer Not Required:{self.model_id}")
            else:
                print("Model Pipline Loading Start...")
                if quantization==True:
                    config=self.quantization(load_in_4bit,load_in_8bit,bnb_4bit_quant_type,bnb_4bit_compute_dtype,bnb_4bit_use_double_quant)
                    model=self.model_initialize(config)
                else:
                    model=self.model_initialize()

                tokenizer=self.tokenizer_initialize()
                pipe = pipeline(task=task,
                        model=model,
                        tokenizer = tokenizer,
                        max_new_tokens = max_new_tokens,
                        temperature = temperature,
                        torch_dtype=torch_dtype,
                        device_map={"":0})
                print("Model Pipline Loaded Sucessfully...")
                logging.info(f"Pipeline Loaded Sucessfully...:{self.model_id}")
                return pipe
        except:
            pass

if __name__=="__main__":
    obj=LLMModel(model_id="TheBloke/Llama-2-7B-Chat-GGML",model_basename="llama-2-7b-chat.ggmlv3.q2_K.bin")
    # llm_pipe=obj.model_pipeline(load_in_4bit=True, 
    #                             bnb_4bit_quant_type="nf4",
    #                               bnb_4bit_compute_dtype=torch.bfloat16,
    #                               bnb_4bit_use_double_quant=False)
    # llm_pipe_without_q=obj.model_pipeline(torch_dtype=torch.bfloat16)
    obj.model_initialize()
   