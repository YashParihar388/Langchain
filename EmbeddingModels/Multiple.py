from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)
documents = [
    'india is our country',
    'all indians are good',
    'long live Bharat'
]

result = embeddings.embed_documents(documents)

print(result)


