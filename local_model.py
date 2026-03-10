from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
  text:str


os.environ['HF_HOME']='E:/LLM (LANGUAGE MODELS)/HuggingFace_cache'

llm=HuggingFacePipeline.from_model_id(
  model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task="text-generation",
  pipeline_kwargs=dict(
    temperature=0.1,
    max_new_tokens=40
  )
)

model=ChatHuggingFace(llm=llm)

@app.post('/generate')
def generate_text(prompt:Prompt):
  res=model.invoke(prompt.text)
  cleaned_response = res.content
  import re
  cleaned_response = re.sub(r'<\|user\|>.*?</s>\s*', '', cleaned_response, flags=re.DOTALL)
  cleaned_response = re.sub(r'<\|assistant\|>\s*', '', cleaned_response)
  cleaned_response = re.sub(r'</s>\s*$', '', cleaned_response)
  
  cleaned_response = re.sub(r'\n\s*References?:.*$', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
  cleaned_response = re.sub(r'\n\s*Capital of India can also be found in:.*$', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
  cleaned_response = re.sub(r'\n\s*Sources?:.*$', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
  
  cleaned_response = cleaned_response.strip()
  return {"response":cleaned_response}