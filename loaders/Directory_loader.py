from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

text = DirectoryLoader(
    path='loaders',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

loaded_text=text.lazy_load()

for load in loaded_text:
    print(load.metadata)