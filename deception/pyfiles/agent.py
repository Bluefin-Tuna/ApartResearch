from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
import envkey
import re
import os

envkey.load()

CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

agent_gpt_0 = ChatOpenAI(model="gpt-4-turbo", api_key=os.environ['OPENAI_API_KEY'], cache=False)
agent_claude_0 = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'], cache=False)
agent_mixstral_0 = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHERAI_API_KEY'], cache=False)

def get_draw_card_fn(agent, prompt):
    def func(game_state):
        return agent.invoke(prompt.format(game_state=game_state))
    return func

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