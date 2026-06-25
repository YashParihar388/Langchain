from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

test = """
The old lighthouse stood quietly on the edge of the cliff,
watching waves crash against the rocks below. Every evening, 
the sky turned shades of orange and purple as seabirds circled overhead.
A narrow path led to the tower, where travelers often stopped to enjoy the
view and listen to the sound of the sea. Despite its age, the lighthouse
remained a symbol of guidance and resilience, shining through storms and 
calm nights alike.
"""

split = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    
)
result = split.split_text(test)

print(result)