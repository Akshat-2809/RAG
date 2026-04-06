from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory ="chroma-db",
)

res = vectorstore.similarity_search("What is Python used for?", k=2)
for x in res:
    print(x.page_content)

retriever = vectorstore.as_retriever()
docs = retriever.invoke('explain deep learning')
for d in docs:
    print(d.page_content)