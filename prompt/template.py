from langchain_core.messages import  HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate 

template = ChatPromptTemplate([
    SystemMessage(content = 'you are a helpful {domain} expert'),
    HumanMessage(content = 'explain in simple terms, what is {topic}')
    ])

response = template.invoke({'domain':'gaming','topic':'pubg'})

print(response)


