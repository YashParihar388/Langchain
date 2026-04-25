from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import os

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv('GROK_KEY'))

chat_history = [SystemMessage('You are good dental assistant')]

while True:
    
    user_input = input('YOU:')
    chat_history.append(HumanMessage(content = user_input))
    if(user_input == 'exit'):
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)
    
