from langchain_openai import ChatOpenAI
import envkey
import os

envkey.load()

gpt = ChatOpenAI(model="gpt-4", api_key=os.environ['OPENAI_API_KEY'])
