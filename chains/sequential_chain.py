from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

parser = StrOutputParser()

template1 = PromptTemplate(
    template='give report on the {topic}',
    input_variables=['report']
)

template2 = PromptTemplate(
    template='give 5 pointer summary on {topic}',
    input_variables=['topic']
)
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black colour'})

chain.get_graph().print_ascii()
