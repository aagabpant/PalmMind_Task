import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))


def get_pdf_text(pdf_docs):
    extracted_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
    return extracted_text


def get_text_chunks(text, chunk_size=10000):
  chunks = []
  start_index = 0
  while start_index < len(text):
    end_index = min(start_index + chunk_size, len(text))
    chunk = text[start_index:end_index]
    chunks.append(chunk)
    start_index = end_index
  return chunks


def get_vector_store(text_chunks):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  try:
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store saved successfully!")
  except Exception as e:
    st.error(f"Error creating vector store: {e}")




def get_conversational_chain():
    prompt_template = """
    **Answer the question in detail, using the provided documents and your knowledge.** 
    * If the answer can be directly found in the documents, provide it with context from the documents.
    * If the answer cannot be directly found in the documents, but can be inferred based on your knowledge and the documents, provide your best answer and explain how you arrived at it. 
    * If the answer cannot be found or inferred, inform the user that the answer is not available in the documents. 

    **Context:**

    * Documents: {context}

    **Question:**

    {question}

    **Answer:**
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)


    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return


    if "call me" in user_question.lower():
                collect_user_info(user_question)
                return

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_history = st.session_state["chat_history"]


    chat_history.append({"question": user_question, "answer": response["output_text"]})
    st.session_state["chat_history"] = chat_history


    st.subheader("Chat With Our Bot")
    for turn in chat_history:
        st.write(f"You: {turn['question']}")
        st.write(f"Bot: {turn['answer']}")





def collect_user_info(user_question):
  st.write("I'd be happy to call you! To schedule a call, please provide your name, phone number, and email address.")

  name = st.text_input("Your Name:")
  phone_number = st.text_input("Phone Number:")
  email = st.text_input("Email Address:")

  with st.container():
      submitted = st.button("Send for Call Request")

  if submitted and name and phone_number and email:

      # ( send it to a database or trigger an email notification anything you like)
      st.success(f"Thank you! Your call request has been received  {name}. We will contact you shortly.")
  else:
      if not submitted:
          st.warning("Please fill in all required fields and click 'Submit Call Request'.")





def main():
    """This is the main function for the PDF chat application"""

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon=":file_pdf:",
        layout="wide",
    )


    st.header("ChatBot PalmMind")

    user_question = st.text_input("Ask a Question", key="user_question", placeholder="Enter your question here...")

    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Upload Documents")
        pdf_docs = st.file_uploader("Select PDFs", accept_multiple_files=True, type="pdf")

        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")


if __name__ == "__main__":
    main()
