from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
import envkey
import re
import os

envkey.load()

CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

gpt = ChatOpenAI(model="gpt-4-turbo", api_key=os.environ['OPENAI_API_KEY'], cache=False)
claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'], cache=False)
mixstral = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHERAI_API_KEY'], cache=False)

def parse_response(response):
    strout = response.content \
                .strip() \
                .lower() \
                .replace("'", "") \
                .replace('`', "") \
                .replace('"', "")
    str_cards = strout.split("\n")[0]
    li_cards = re.split(",\s?", str_cards)
    if li_cards[0] == "random":
        return True, li_cards
    is_valid = all([card in CARDS for card in li_cards])
    return is_valid, li_cards