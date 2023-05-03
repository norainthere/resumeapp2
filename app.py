import os
import streamlit as st
import openai
from langchain.llms import OpenAI
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Fetch OpenAi API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set up OpenAI Client
openai.api_key = st.secrets["openai"][openai_api_key]

# Set up OpenAI Playground Trained Model
playground_model = 

#Initialize the prompt for
#Set prompt
prompt=

# App framework
st.title()
#Generate Text
prompt = st.text_input()
if prompt:
    prompt =
    response = openai.Completion.create(
        engine=openai_model
        prompt=prompt
        max_tokens=2000
    )
    generated_text = response.choices[0].text
    formatted_text = generated_text.replace("\n", "\n-")
    st.markdown(f"****\n\n- {formatted_text})