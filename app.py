import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        reader=PdfReader(pdf)
        for page in reader.pages:
            text+=page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/asasfjkhdsfjk", model_kwargs={"temperature":jdf, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
def get_vector_store(text_chunks):
   embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
   vector_store=FAISS.from_texts(texts=text_chunks,embeddings=embeddings)
   return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title='My Streamlit App', page_icon=':books:')
    
    st.header("Textchatapp :books: ")
    st.text_input("Enter your question")

    with st.sidebar:
        st.subheader("your files")
        pdf_docs=st.file_uploader("Upload your file", accept_multiple_files=True)
        if st.button("Submit"):
           with st.spinner("Processing your files"):
                #getting the large string from all pdfs
                raw_text=get_pdf_text(pdf_docs)
                
                #splitting the text into chunks
                text_chunks=get_text_chunks(raw_text)
                #st.write(text_chunks)
                #getting the vector store
                vector_store=get_vector_store(text_chunks)
                 # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)





if __name__=='__main__':
    main()