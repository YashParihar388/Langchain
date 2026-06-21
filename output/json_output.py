from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv("GROK_KEY"))

parser = JsonOutputParser()

template = PromptTemplate(
    template = "give me the 5 facts about {topic} \n {instruction}",
    input_variables=["topic"],
    partial_variables={"instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic":"black hole"})

print(result)