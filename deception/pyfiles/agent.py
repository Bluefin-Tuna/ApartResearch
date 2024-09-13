from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_together import ChatTogether
import envkey
import re
import os

envkey.load()

CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

agent_gpt_0 = ChatOpenAI(model="gpt-4o-2024-08-06", api_key=os.environ['OPENAI_API_KEY'], temperature=0, cache=False)
agent_gpt_5 = ChatOpenAI(model="gpt-4o-2024-08-06", api_key=os.environ['OPENAI_API_KEY'], temperature=0.5, cache=False)
agent_claude_0 = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'], temperature=0, cache=False)
agent_claude_5 = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ['ANTHROPIC_API_KEY'], temperature=0.5, cache=False)
# agent_mixstral_0 = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHERAI_API_KEY'], cache=False)

def parse_response(response):
    pattern = r'\b([2-9]|10|ace|queen|jack|king)\b'
    card = re.findall(pattern, response.content, re.IGNORECASE)
    if not card:
        return False, None
    str_card = card[0].lower().strip()
    is_valid = str_card in CARDS
    if not is_valid:
        return is_valid, None
    return is_valid, str_card

def get_draw_card_fn(agent, prompt):
    def func(game_state):
        output = agent.invoke(prompt.format(game_state=game_state))
        return parse_response(output)
    return func