from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv("GROK_KEY"))

class Person(BaseModel):
    name: str = Field(description='name of the person')
    age: int = Field(gt=0,description='age of that person')
    city: str = Field(description='city that person live in')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
     template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'Australia'})

print(result)