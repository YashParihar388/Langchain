from langchain_groq import ChatGroq
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

schema =[
    ResponseSchema(name='fact_1',description='fact 1 about this topic'),
    ResponseSchema(name='fact_2',description='fact 2 about this topic'),
    ResponseSchema(name='fact_3',description='fact 3 about this topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template='give 3 fact about {topic} \n{instruction}',
    input_variables=['topic'],
    partial_variables= {"instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)