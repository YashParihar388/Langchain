from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal
import os
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROK_KEY"))
class Review(TypedDict):
    summary:Annotated[str,"give summary of whole in 2 lines"]
    sentiment:Annotated[Literal["positive","negative"],"give the sentiment means emotion of the user"]

structured_output=model.with_structured_output(Review)
    
result = structured_output.invoke("this is the samsung phone and it is very amazing im happy to purchase this particular item.But at the same time it is very bad also.But i would suggest that it is good overall")


print(result['sentiment'])
