from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b',api_key =os.getenv("GROK_KEY") )

message = [SystemMessage(content = 'you are a helpfull assistant'),
           HumanMessage(content='tell me about langchain')]

result = model.invoke(message)

message.append(AIMessage(content = result.content))

print(message)