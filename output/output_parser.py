from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

template1 = PromptTemplate(
    template='write a short  report on {topic}',
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='write a 5 line summary on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
