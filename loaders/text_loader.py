from langchain_community.document_loaders import TextLoader;
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

template1 = PromptTemplate(
    template='write a appropriate reply for this  {message}',
    input_variables=['message']
)
parser = StrOutputParser()

chain = template1 | model | parser

text = TextLoader('loaders/cricket.txt')

loaded_text = text.load()

chain = template1 | model | parser

print(chain.invoke({"message":loaded_text[0].page_content}))