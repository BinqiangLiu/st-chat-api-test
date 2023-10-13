from flask import Flask, request, jsonify
from pathlib import Path
import streamlit as st
from streamlit_chat import message
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests# Internal usage
import os
from dotenv import load_dotenv
from time import sleep
import uuid
import sys
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, LLMChain

st.set_page_config(page_title="AI Chatbot 100% Free", layout="wide")
st.write('完全开源免费的AI智能聊天助手 | Absolute Free & Opensouce AI Chatbot')

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id=os.getenv("repo_id")
port=os.getenv("port")

av_us = '🧑'
av_ass = '🤖'

#def starchat(model, myprompt, my_prompt_template):
def starchat(repo_id, myprompt): 
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"min_length":1024,
                                       "max_new_tokens":5632, "do_sample":True,
                                       "temperature":0.1,
                                       "top_k":50,
                                       "top_p":0.95, "eos_token_id":49155})     
    my_prompt_template = """
    <<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    In each conversation, question is placed after [INST] while your answer should be placed after [/INST]. By looking [INST] and [/INST], you must consider multi-turn conversations saved in {contexts}.<</SYS>>
    [INST] {myprompt} [/INST] 
    assistant:
    """    
    template = my_prompt_template    
    prompt = PromptTemplate(template=template, input_variables=["contexts", "myprompt"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_reply = llm_chain.run({'contexts': contexts, 'myprompt': myprompt})  
    reply = llm_reply.partition('<|end|>')[0]    
    return reply

if "file_name" not in st.session_state:
    st.session_state["file_name"] = str(uuid.uuid4()) + ".txt"    

def writehistory(text):           
    with open(st.session_state["file_name"], 'a+') as f:
        f.write(text)
        f.write('\n')
        f.seek(0) 
        contexts = f.read()        
    return contexts

if "messages" not in st.session_state:
    st.session_state.messages = []
#for message in st.session_state.messages:
#   if message["role"] == "user":
#      with st.chat_message(message["role"],avatar=av_us):
#      with st.chat_message(message["role"]):                  
#           st.markdown(message["content"])           
#   else:
#       with st.chat_message(message["role"],avatar=av_ass):
#       with st.chat_message(message["role"]):                   
#           st.markdown(message["content"])           

temp_myprompt = st.chat_input("Enter your question here.")

app = Flask(__name__)
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    #model = data['model']    
    myprompt = data['user_question']
    temp_myprompt=myprompt
#if myprompt := st.chat_input("Enter your question here."):    
    st.session_state.messages.append({"role": "user", "content": myprompt}) 
    #with st.spinner("AI Thinking..."):   
#    with st.chat_message("user", avatar=av_us):
    with st.chat_message("user"):        
        #st.markdown(myprompt)        
        usertext = f"user: {myprompt}"              
        contexts = writehistory(usertext)          
    with st.chat_message("assistant"):
        #message_placeholder = st.empty()
        full_response = ""
        res = starchat(repo_id, myprompt)
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            #message_placeholder.markdown(full_response + "|")
            #sleep(0.1)                       
        #message_placeholder.markdown(full_response)            
        asstext = f"assistant: {full_response}"  
        contexts = writehistory(asstext)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    return jsonify({"response": "API TEST"})
#with st.spinner("AI Thinking..."):   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
#    app.run(host='0.0.0.0')
