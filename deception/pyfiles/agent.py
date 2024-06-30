from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
import envkey
from blackjack import CARDS
import os

envkey.load()

gpt = ChatOpenAI(model="gpt-4-turbo", api_key=os.environ['OPENAI_API_KEY'], cache=False)
claude = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'], cache=False)
mixstral = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHERAI_API_KEY'], cache=False)

def parse_response(response):
    strout = response.content.strip()
    str_cards = strout.split("\n")[0]
    li_cards = str_cards.split(",")
    if li_cards[0] == "random":
        return True, li_cards
    is_valid = all([card in CARDS for card in li_cards])
    return is_valid, li_cards