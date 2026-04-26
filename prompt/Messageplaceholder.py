from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate

chat =ChatPromptTemplate([
    ('system','you are a helpfull assistant'),
    MessagesPlaceholder(variable='chat_history'),
    ('human','{query}')
]
    
)
chat_history = []

with open('chat_history.txt') as f:
    chat_history.append(f.readlines())

result = chat.invoke('chat_history':chat_history,'query':'where is my refund')
