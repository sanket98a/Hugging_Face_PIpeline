import os
from qa_bot.ingestion import create_vector_db
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from qa_bot.constant import CHROMA_SETTINGS
import yaml
import torch
from src.model  import LLMModel
from transformers import pipeline,GenerationConfig
from langchain.llms import HuggingFacePipeline
import chainlit as cl
import shutil

# ROOT Dir
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY=os.path.join(ROOT_DIRECTORY,'qa_bot/DB')
print('*'*50)
print("PERSIST_DIRECTORY",PERSIST_DIRECTORY)
print('*'*50)

data_file=os.listdir(os.path.join(ROOT_DIRECTORY,'qa_bot/data'))
print('*'*50)
print("data_file",data_file)
print('*'*50)

with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# check the device 
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

# Check any file present in data folder if present do embedding and delete the file
if len(data_file)>=1:
    # check prev DB, if present remove it
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Prev. Embedding Remove")
        create_vector_db()
        print("Embedding Stored Succesfully..")
        # check New Embedding Store or Not
        if os.path.exists(PERSIST_DIRECTORY):
            for file in data_file:
                os.remove(os.path.join('qa_bot/data',file))
                print(f"File removed :: {file}")
else:
    print("File Not Present in dir")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template.strip()

system_prompt = "Use the following pieces of context to answer the question at the end. If the answer cannot be found, respond with 'The answer is not available in the given data'.\ncontext:{context}"
instruction = """user_question: {question}"""
prompt_template=get_prompt(instruction,system_prompt)
print(prompt_template)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm(model_id,model_subname=None,max_length=512,temperature=0.7):
    if ".ggml" in model_subname:
        model_loading=LLMModel(model_id=model_id,model_basename=model_subname)
        model=model_loading.model_initialize()
        print("Model Loaded")
        return model
    else:
        # Others Model Loading
        model_loading=LLMModel(model_id=model_id,model_basename=model_subname)
        model=model_loading.model_initialize()
        tokenizer=model_loading.tokenizer_initialize()
        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)
        # LLM Pipeline added
        pipe=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        # pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        max_length=max_length,
        generation_config=generation_config)
        # Load model
        hug_model=HuggingFacePipeline(pipeline=pipe)
        return hug_model

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name=config['EMBEDDING_MODEL_NAME'],
                                       model_kwargs={'device': device_type})
    
    # embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,
    #                                    model_kwargs={'device': 'cpu'})
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    # model_subname="llama-Llama-2-7B-Chat-GGML"
    # model_id="TheBloke/2-7b-chat.ggmlv3.q2_K.bin"
    ## GPTQ Quantized Model
    model_id="TheBloke/Llama-2-7b-Chat-GPTQ"
    model_subname="gptq_model-4bit-128g.safetensors"
    llm = load_llm(model_id=model_id,model_subname=model_subname)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Affine Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # print('*'*70)
    # print("ANSWER::",answer)
    # print('*'*70)
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()



