from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import PathConfig
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate


def ask_to_llm(query: str, sentiment_label: int, topic: int, num_k: int = 5):
    llm = ChatOllama(model="llama3", temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=str(PathConfig.VECTORSTORE_PATH),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {
                "$and": [{"sentiment_label": sentiment_label}, {"topic": topic}]
            },
            "k": num_k,
        }
    )
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = retrieval_chain.invoke({"input": query})

    print("### AI Answer ###")
    print(response["answer"])
    print(response["context"])
