import streamlit as st
import os
from huggingface_hub import hf_hub_download

with st.sidebar:
    model_path=r"C:\Users\sanket\.cache\huggingface\hub"
    models=[model.split('--')[-1] for model in os.listdir(model_path) if 'model' in model]
    st.selectbox("Select Model",models)
    flag=st.radio("Model not found",[False,True],horizontal=True)
    quantized=st.radio("Model Type",['Quantized','Normal'],horizontal=True)
    
    if quantized=='Quantized' and flag==False:
        st.selectbox("Select Model BaseName",["llama_7b_quantized"])
    if flag==True:
        model_id=st.text_input("model_id")
        if model_id and quantized=='Quantized':
            model_basename=st.text_input("model_basename")
            # model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
        elif model_id:
            # model_path = hf_hub_download(repo_id=model_id)
            pass
    else:
        pass
    


    
    





