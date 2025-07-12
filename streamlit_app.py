import streamlit as st
import requests
import os
from dotenv import load_dotenv

BACKEND_URL = "http://localhost:8000"

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/abdul-rehman-57a192241/"
    github_profile_link = "https://github.com/AbdulRehman5592/s"

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
    
    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )


    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")
        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.session_id = None
            # Call backend reset
            if st.session_state.session_id:
                requests.post(f"{BACKEND_URL}/reset/", data={"session_id": st.session_state.session_id})
        elif clear_button:
            if st.session_state.conversation_history:
                st.warning("The previous query will be discarded.")
                st.session_state.conversation_history.pop()
            else:
                st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    files = [("files", (pdf.name, pdf, pdf.type)) for pdf in pdf_docs]
                    data = {}
                    if st.session_state.session_id:
                        data["session_id"] = st.session_state.session_id
                    response = requests.post(f"{BACKEND_URL}/upload_pdfs/", files=files, data=data)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.session_id = result["session_id"]
                        st.success(f"Uploaded {len(pdf_docs)} PDFs. Chunks: {result['chunks']}")
                    else:
                        try:
                            error_msg = response.json().get("error", "Failed to upload PDFs.")
                        except Exception:
                            error_msg = response.text or "Failed to upload PDFs. (Non-JSON response)"
                        st.error(error_msg)
            else:
                st.warning("Please upload PDF files before processing.")

        # --- New: Encode PDFs to base64 and display ---
        if pdf_docs:
            if st.button("Encode PDFs to Base64"):
                import base64
                base64_results = []
                for pdf in pdf_docs:
                    pdf_bytes = pdf.read()
                    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    base64_results.append({"filename": pdf.name, "base64": b64})
                for result in base64_results:
                    with st.expander(f"Base64 for {result['filename']}"):
                        st.text_area("Base64 String", result["base64"], height=150)

    user_question = st.text_input("Ask a Question from the PDF Files")
    if st.button("Ask"):
        if not user_question:
            st.warning("Please enter a question.")
        elif not st.session_state.session_id:
            st.warning("Please upload and process PDFs first.")
        else:
            with st.spinner("Getting answer..."):
                data = {"query": user_question, "session_id": st.session_state.session_id}
                response = requests.post(f"{BACKEND_URL}/chat/", data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.conversation_history.append({
                        "question": user_question,
                        "answer": result["answer"],
                        "timestamp": result["timestamp"]
                    })
                    st.success(result["answer"])
                else:
                    st.error(response.json().get("error", "Failed to get answer."))

    if st.session_state.session_id:
        if st.button("Show History"):
            response = requests.get(f"{BACKEND_URL}/history/", params={"session_id": st.session_state.session_id})
            if response.status_code == 200:
                history = response.json().get("history", [])
                st.write("## Conversation History")
                for item in history:
                    st.write(f"**Q:** {item['question']}")
                    st.write(f"**A:** {item['answer']}")
                    st.write(f"_Time:_ {item['timestamp']}")
                    st.write("---")
            else:
                st.error("Failed to fetch history.")

if __name__ == "__main__":
    main() 