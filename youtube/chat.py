from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda , RunnablePassthrough

load_dotenv()

video_id = "aNk0km8h0NE"

try:
    yt_api = YouTubeTranscriptApi()
    transcript =yt_api.fetch(video_id,languages=["en"])
    text = " ".join(chunk.text for chunk in transcript)
    
    # for item in transcript:
    #    print(item.text)

except TranscriptsDisabled:
    print("captions are turned off for this video")
    
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([text])

embeddings = CohereEmbeddings(model="embed-english-v3.0")
vector_store = FAISS.from_documents(chunks,embeddings)

prompt = PromptTemplate(
    template ="""
         You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
    
    {context}
    question : {question}
    """,
    input_variables=['context', 'question']
    
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# question = 'is tcs nqt easy?'
# documents = retriever.invoke(question)

# context_text = "/n/n".join(doc.page_content for doc in documents)

# final_prompt = prompt.invoke({"context":context_text,"question":question})

llm = ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv("GROK_KEY"))

# result = llm.invoke(final_prompt)

# print(result.content)


def format(documents):
    text = "\n\n".join(doc.page_content for doc in documents)
    return text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser 

result = main_chain.invoke('can you summarize this?')

print(result)
