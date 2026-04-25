from langchain_cohere import CohereEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding=CohereEmbeddings(model = 'embed-english-v3.0')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about kohli'

docu_embed = embedding.embed_documents(documents)
query_embed = embedding.embed_query(query)

scores = cosine_similarity(docu_embed,[query_embed])

refine = list(enumerate(scores))

print(scores)



