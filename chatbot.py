import os
from qa_bot.ingestion import create_vector_db

# Check DB Present in current location
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

DB_Flag=os.path.exists(os.path.join(ROOT_DIRECTORY,'DB'))

data_file=os.listdir(os.path.join(ROOT_DIRECTORY,'qa_bot/data'))


if DB_Flag==False and len(data_file)>=1:
    create_vector_db()
    print("Embedding Stored Succesfully..")
else:
    print("File Not Present in dir")


