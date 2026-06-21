from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.runnables import  RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

class Feedback(BaseModel):
    sentiment: Literal["negative","positive"] = Field(description='this is the overall sentiment of the feedback')
    
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template = 'classify the catgeory as positive or negative on the basis of following feedback \n {feedback} , {instruction}',
    input_variables=['feedback'],
    partial_variables={"instruction":parser2.get_format_instructions()}
)


parser1 = StrOutputParser()

template2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)


template3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)
chain_classifier = template1 | model | parser2

branch = RunnableBranch(
    (lambda x:x.sentiment == 'positive' , template2 | model | parser1),
    (lambda x:x.sentiment == 'negative' , template3 | model | parser1),
    RunnableLambda(lambda x:'coud not find sentiment')
)

chain = chain_classifier | branch

result = chain.invoke({"feedback":"this smart phone is very bad"})

chain.get_graph().print_ascii()





