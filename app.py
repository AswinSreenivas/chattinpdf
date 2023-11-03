import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from htmltemplates import user_template, bot_template

# User profiles (temporary storage in memory)
user_profiles = {}

class ChatOpenAI:
    pass  # Define the ChatOpenAI class or import it from the correct module

class ConversationBufferMemory:
    pass  # Define the ConversationBufferMemory class or import it from the correct module

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
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
    llm = ChatOpenAI()  # Initialize ChatOpenAI class (if not imported)
    memory = ConversationBufferMemory()  # Initialize ConversationBufferMemory class (if not imported)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if 'user_id' not in st.session_state:
        st.session_state.user_id = st.text_input("Enter your user ID")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    response = st.session_state.conversation({'user_id': st.session_state.user_id, 'question': user_question})
    st.session_state.chat_history.append(response['chat_history'])

    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title='My Streamlit App', page_icon=':books:')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    st.header("Textchatapp :books: ")
    user_question = st.text_input("Enter your question")

    with st.sidebar:
        st.subheader("Your files")
        pdf_docs = st.file_uploader("Upload your file", accept_multiple_files=True)
        if st.button("Submit") and pdf_docs:
            with st.spinner("Processing your files"):
                # Getting the large string from all PDFs
                raw_text = get_pdf_text(pdf_docs)
                # Splitting the text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Getting the vector store
                vector_store = get_vector_store(text_chunks)
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    st.subheader("Chat History")
    for i, chat_entry in enumerate(st.session_state.chat_history):
        st.write(f"User: {chat_entry['user_message']}")
        st.write(f"Bot: {chat_entry['bot_response']}")
        st.write("")

if __name__ == '__main__':
    main()
