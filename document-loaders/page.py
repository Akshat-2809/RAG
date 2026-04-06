from langchain_community.document_loaders import WebBaseLoader

data = WebBaseLoader("https://www.apple.com/macbook-pro/")

docs = data.load()

print(docs)