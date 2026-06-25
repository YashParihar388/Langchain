from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel ,RunnableSequence ,RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

prompt1 = PromptTemplate(
    template ='tell me a good joke about this {topic}',
    input_variables=['topic']
)
prompt2 =PromptTemplate(
    template = 'tell me a joke about this {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()

first_chain = RunnableSequence(prompt1,model,parser)

def count(text):
    return len(text.split())
second_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'length':RunnableLambda(lambda x:len(x.split()))
    }
)

chain = RunnableSequence(first_chain,second_chain)

print(chain.invoke({"topic":"Human"}))