from typing import List
from fastapi import FastAPI, HTTPException
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2Tokenizer,GPT2Model,pipeline
import torch
from src.model  import LLMModel

# app = FastAPI()
app = Flask(__name__)

# Load the LLM model and tokenizer
# model_name = "your_model_name_here"  # Replace with the name of your trained model (e.g., "gpt2", "gpt2-medium", etc.)
# model_path ='gpt2-medium' #"./falcon-7b-instruct/" 
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,)

model_id="TheBloke/Llama-2-7B-Chat-GGML"
model_subname="llama-2-7b-chat.ggmlv3.q2_K.bin"

if ".ggml" in model_subname:
    def generate_text(prompt, max_length=50,temp=0.7):
        model_loading=LLMModel(model_id=model_id,model_basename=model_subname)
        model=model_loading.model_initialize()
        print("Model Loaded")
        return model(prompt)
else:
    model_loading=LLMModel(model_id=model_id,model_basename=model_subname)
    model=model_loading.model_initialize()
    tokenizer=model_loading.tokenizer_initialize()
    # Function to generate text using the LLM model
    def generate_text(prompt, max_length=50,temp=0.7):
        # input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,temperature=temp)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pipline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temp,
        max_length=max_length)
        generated_text=pipline.predict([prompt])
        return generated_text


@app.route("/generate_text/", methods=["POST"])
def generate_text_api():
    data = request.get_json()
    prompts = data.get("prompts")
    max_length = data.get("max_length", 50)
    temp=data.get("temperature",0.7)
    # print(prompts)
    if not prompts:
        return jsonify({"error": "Invalid input. 'prompts' field must be a non-empty list."}), 400

    # generated_texts = [f"Generated: {prompt} (Max Length: {max_length})" for prompt in prompts]
    generated_texts = [generate_text(prompt, max_length,temp) for prompt in prompts]
    return jsonify({"generated_texts": generated_texts})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)



