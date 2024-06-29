import envkey
import os

envkey.load()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
print(OPENAI_API_KEY)