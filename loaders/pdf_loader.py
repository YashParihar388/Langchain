from langchain_community.document_loaders import PyPDFLoader

text = PyPDFLoader('loaders/English practical.pdf')

loaded = text.load()

print(len(loaded))

print(loaded[0].page_content)