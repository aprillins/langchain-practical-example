from langchain_community.document_loaders import DirectoryLoader, CSVLoader

loader = DirectoryLoader("../data", glob="*.csv", loader_cls=CSVLoader)
docs = loader.load()
print(len(docs[0].page_content[:200]))