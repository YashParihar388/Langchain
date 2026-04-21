from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv("GROK_KEY"))

result = llm.invoke('What is my name')

print(result.content)