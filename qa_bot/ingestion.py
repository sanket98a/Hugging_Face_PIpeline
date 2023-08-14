from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma
# from src.logger import logging
import logging
import yaml
import torch
from qa_bot.constant import (
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

# check the device 
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

# Create vector database
def create_vector_db():
    # load the data
    loader = DirectoryLoader(SOURCE_DIRECTORY,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    # path='data/review_english_cleaned.csv'
    # loader=CSVLoader(file_path=file_path,encoding="utf8")
    documents = loader.load()
    
    logging.info(f"Documnet Loaded : {SOURCE_DIRECTORY}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= config['CHUNK_SIZE'],
                                                   chunk_overlap=config['CHUNK_OVERLAP'])
    texts = text_splitter.split_documents(documents)
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    if config['EMBEDDING_TYPE'].lower()=='normal':
        print("HuggingFaceEmbeddings")
        embeddings = HuggingFaceEmbeddings(model_name=config['EMBEDDING_MODEL_NAME'],
                                        model_kwargs={"device": device_type}) 
        logging.info(f"HuggingFaceEmbeddings the Document : {config['EMBEDDING_MODEL_NAME']}")
    else:
        print("HuggingFaceInstructEmbeddings")
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=config['EMBEDDING_MODEL_NAME'],
            model_kwargs={"device": device_type})
        logging.info(f"HuggingFaceInstructEmbeddings the Document : {config['EMBEDDING_MODEL_NAME']}")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    logging.info(f"Embedding Stored in Chromadb : {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    create_vector_db()

