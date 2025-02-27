import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer short.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = "documents"

embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:14b")


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    return text_splitter.split_documents(documents)


def index_docs(documents):
    vector_store.add_documents(documents)


def retrieve_docs(query, k=5):
    return vector_store.similarity_search(query, k=k)


def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})


def main():
    for file in os.listdir(pdfs_directory):
        if file.endswith(".pdf"):
            print(f"Indexing {file}")
            documents = load_pdf(os.path.join(pdfs_directory, file))
            documents = split_text(documents)
            index_docs(documents)

    while True:
        query = input("Enter your question: ")
        docs = retrieve_docs(query)
        print(f"Retrieved chunks: {len(docs)}")
        answer = answer_question(query, docs)
        print(answer)


if __name__ == "__main__":
    main()
