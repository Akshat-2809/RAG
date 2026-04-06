from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs= 2,
    load_all_available_meta=True
)

docs = retriever.invoke('large language models')

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print("title:", doc.metadata.get('title'))
    print("Authors:", doc.metadata.get('authors'))
    print("Summary:", doc.page_content) 