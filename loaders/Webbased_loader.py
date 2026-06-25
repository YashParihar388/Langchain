from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/vivo-y22-metaverse-green-128-gb/p/itmb457f45cfdbeb'

text = WebBaseLoader(url)

loaded_text = text.load()

print(len(loaded_text))