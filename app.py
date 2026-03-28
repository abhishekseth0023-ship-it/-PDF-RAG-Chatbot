import streamlit as st
from RAG_Pipeline import build_vectorstore, ask_question

st.title("PDF RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Upload a PDF", type="pdf",accept_multiple_files=True)

if uploaded_files:

    pdf_paths = []

    for file in uploaded_files:

        pdf_path = f"data/{file.name}"

        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())

        pdf_paths.append(pdf_path)

    st.success("PDFs uploaded successfully!")

    with st.spinner("Processing PDFs..."):
        build_vectorstore(pdf_paths)


    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the PDF")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        answer, pages = ask_question(question)

        with st.chat_message("assistant"):
            st.markdown(answer)

            unique_pages = sorted(set(pages))

            st.write("**Sources:**")
            for p in unique_pages:
                st.write(f"Page {p}")

        st.session_state.messages.append({"role": "assistant", "content": answer})