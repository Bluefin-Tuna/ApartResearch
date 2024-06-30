from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
import envkey
import os

envkey.load()

gpt = ChatOpenAI(model="gpt-4", api_key=os.environ['OPENAI_API_KEY'])
claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'])
mixstral = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHERAI_API_KEY'])
llama = ChatTogether(model="meta-llama/Meta-Llama-3-70B", api_key=os.environ['TOGETHERAI_API_KEY'])