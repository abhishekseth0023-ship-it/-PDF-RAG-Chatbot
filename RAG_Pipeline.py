from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import OllamaLLM

def build_vectorstore(pdf_paths):

    all_docs=[]

    for pdf in pdf_paths:
        loader=PyMuPDFLoader(pdf)
        docs=loader.load()
        all_docs.extend(docs)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks=splitter.split_documents(all_docs)

    embeddings=HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore=FAISS.from_documents(chunks,embeddings)

    vectorstore.save_local("vectorstore")

    return vectorstore

def load_vectorstore():
    
    embeddings=HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore=FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

vectorstore=load_vectorstore()

llm=OllamaLLM(model="phi3",temperature=0)

def ask_question(question):

    retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":3})

    docs=retriever.invoke(question)

    context="\n".join([doc.page_content for doc in docs])

    pages=[doc.metadata.get("page","Unknown") for doc in docs]

    prompt = f"""
    You are an AI assistant that answers questions strictly using the provided document context.

    Guidelines:
    - Only use the information inside the context.
    - Do not rely on external knowledge.
    - If the context does not contain the answer, say:
    "I could not find the answer in the document."
    - Provide a clear and concise explanation.

    Context:
    {context}

    Question:
    {question}
    """

    response=llm.invoke(prompt)

    return response,pages
