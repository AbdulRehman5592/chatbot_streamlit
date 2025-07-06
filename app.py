import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64

import os

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2.errors import PdfReadError



from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv



from datetime import datetime

import time

# Create a placeholder
message_container = st.empty()

# Animate the message appearing
with message_container.container():
    st.markdown("Your message here")
    time.sleep(0.1)  # Small delay for animation effect


def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        try:
            pdf.seek(0)  # Required for Streamlit's uploaded files
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    print(f"[DEBUG] Extracted text length: {len(text)}")
        except PdfReadError:
            print(f"⚠️ Skipping corrupted PDF: {getattr(pdf, 'name', 'Unknown file')}")
        except Exception as e:
            print(f"❌ Unexpected error reading {getattr(pdf, 'name', 'Unknown file')}: {e}")
    
    return text

def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(f"[DEBUG] Number of text chunks: {len(chunks)}")
    return chunks

def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if not text_chunks:
        raise ValueError("No text chunks provided.")

    # embed_documents generates the actual embeddings list
    embedded_vectors = embedding_model.embed_documents(text_chunks)

    if not embedded_vectors:
        raise ValueError("Embedding model returned no vectors.")

    # Check length match
    if len(embedded_vectors) != len(text_chunks):
        raise ValueError("Mismatch between text_chunks and embedded_vectors")

    # FAISS expects an Embeddings object, not a list of vectors, so we do:
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    print(f"[DEBUG] get_conversational_chain called with model_name={model_name}")
    if model_name == "Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        print(f"[DEBUG] get_conversational_chain returning chain: {chain}")
        return chain
    print("[DEBUG] get_conversational_chain did not match any model_name, returning None")
    return None

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    user_question_output = ""
    response_output = ""
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
        if chain is None:
            st.error("Failed to initialize the conversational chain.")
            print("[DEBUG] get_conversational_chain returned None")
            return
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        # conversation_history.append((user_question_output, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    # Kullanıcının sorduğu soruyu ve cevabı bir banner olarak ekleyelim
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(to bottom, skyblue, purple);
        background-attachment: fixed;
    }}
    @keyframes slideInFromBottom {{
        from {{
            transform: translateY(50px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    .chat-message {{
        animation: slideInFromBottom 0.5s ease-out;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        background-color: rgba(255, 255, 255, 0.05);
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
    }}
    .chat-message.user {{
        background-color: rgba(43, 49, 62, 0.7);
    }}
    .chat-message.bot {{
        background-color: rgba(71, 80, 99, 0.7);
    }}
    .chat-message .avatar {{
        width: 15%;
    }}
    .chat-message .avatar img {{
        max-width: 60px;
        max-height: 60px;
        border-radius: 50%;
        object-fit: cover;
    }}
    .chat-message .message {{
        width: 85%;
        padding: 0 1.5rem;
        color: #fff;
        font-size: 1rem;
    }}
    .chat-message .info {{
        font-size: 0.8rem;
        margin-top: 0.5rem;
        color: #ccc;
    }}
    </style>

    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
        </div>    
        <div class="message">{user_question_output}</div>
    </div>
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
        </div>
        <div class="message">{response_output}</div>
    </div>
    """,
    unsafe_allow_html=True
)

            # <div class="info" style="margin-left: 20px;">Timestamp: {datetime.now()}</div>
            # <div class="info" style="margin-left: 20px;">PDF Name: {", ".join(pdf_names)}</div>
    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1 :
        last_item = conversation_history[-1]  # Son öğeyi al
        conversation_history.remove(last_item) 
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
                # <div class="info" style="margin-left: 20px;">Timestamp: {timestamp}</div>
                # <div class="info" style="margin-left: 20px;">PDF Name: {pdf_name}</div>

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])  # type: ignore

        # df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.snow()
    st.markdown(
    """
    <script>
    document.body.classList.add('chat-active');
    </script>
    """,
    unsafe_allow_html=True
)
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")
    
    st.markdown(
    """
    <style>
    @keyframes slideInFromLeft {
        0% {
            transform: translateX(-100%);
            opacity: 0;
        }
        70% {
            transform: translateX(10px);
            opacity: 1;
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .stApp {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }
    
    
    /* Background behind sidebar during animation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }

    /* Also style the sidebar content area */
    section[data-testid="stSidebar"] .css-1d391kg {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }

    /* Header with green gradient initially (matching theme) */
    header[data-testid="stHeader"],
    .stApp header,
    header {
        background: linear-gradient(to bottom, #b9f6ca, #43a047) !important;
        background-attachment: fixed !important;
        transition: background 0.5s ease;
    }

    /* Header changes to match screen gradient when chat messages are present */
    .stApp:has(.chat-message) header[data-testid="stHeader"],
    .stApp:has(.chat-message) header {
        background: linear-gradient(to bottom, skyblue, purple) !important;
        background-attachment: fixed !important;
    }

    /* Alternative approach using JavaScript-like state */
    .stApp .chat-message ~ header[data-testid="stHeader"] {
        background: linear-gradient(to bottom, #e1bee7, #7b1fa2) !important;
        background-attachment: fixed !important;
    }

    /* Alternative purple gradients you can try: */
    /* Light to medium purple */
    /* background: linear-gradient(to bottom, #f3e5f5, #9c27b0); */

    /* Medium to dark purple */
    /* background: linear-gradient(to bottom, #ce93d8, #4a148c); */

    /* Purple to blue */
    /* background: linear-gradient(to bottom, #e1bee7, #3f51b5); */

    /* Header background when sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }

    /* Alternative approach - style the header directly */
    header[data-testid="stHeader"] {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }

    /* Also ensure the main content area maintains theme */
    .main .block-container {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
    }

    /* Sidebar background with animation */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(to bottom, #b9f6ca, #43a047);
        background-attachment: fixed;
        border-right: 4px solid #388e3c;
        box-shadow: 2px 0 8px rgba(67, 160, 71, 0.15);
        animation: slideInFromLeft 1.5s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    linkedin_profile_link = "https://www.linkedin.com/in/abdul-rehman-57a192241/"
    github_profile_link = "https://github.com/AbdulRehman5592/s"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )



    model_name = st.sidebar.radio("Select the Model:", ( "Google AI"))

    load_dotenv("keys.env")
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("API key not found. Please create a .env file with your GOOGLE_API_KEY.")
        return

   
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            
            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")




        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()