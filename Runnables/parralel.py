from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel ,RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))

prompt1 = PromptTemplate(
    template ='tell me in very short  about this {topic}',
    input_variables=['topic']
)
prompt2 =PromptTemplate(
    template = 'tell me a joke about this {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()

chain = RunnableParallel({
    'report': RunnableSequence(prompt1,model,parser),
    'joke':RunnableSequence(prompt2,model,parser)
})

print(chain.invoke({'topic':'AI'}))

