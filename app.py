import os
from typing import List, Dict
from dataclasses import dataclass
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

@dataclass
class Config:
    chunk_size: int = 10000
    chunk_overlap: int = 1000
    temperature: float = 0.3
    model_name: str = "gemini-pro"
    embedding_model: str = "models/embedding-001"
    vector_store_path: str = "faiss_index"

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_docs) -> str:
        """Extract text from multiple PDF documents."""
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                st.error(f"Error processing PDF {pdf.name}: {str(e)}")
        return text

    @staticmethod
    def create_chunks(text: str, config: Config) -> List[str]:
        """Split text into manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        return splitter.split_text(text)

class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model
        )

    def create_and_save(self, chunks: List[str]):
        """Create and save vector store from text chunks."""
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local(self.config.vector_store_path)

    def load(self):
        """Load existing vector store."""
        return FAISS.load_local(
            self.config.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

class CivilEngineerChain:
    def __init__(self, config: Config):
        self.config = config
        self.prompt_template = """
        You are an expert AI Civil Engineering Assistant with comprehensive knowledge across multiple domains. Your expertise includes:

        Technical Domains:
        1. Structural Engineering
           - Advanced structural analysis and design
           - Seismic design considerations
           - Foundation engineering
           - Construction materials and technology

        2. Transportation Engineering
           - Highway and railway design
           - Traffic flow analysis
           - Transportation planning
           - Infrastructure maintenance

        3. Environmental Engineering
           - Sustainable design practices
           - Water resources management
           - Environmental impact assessment
           - Waste management systems

        4. Construction Management
           - Project planning and scheduling
           - Cost estimation and control
           - Quality management
           - Risk assessment and mitigation

        5. Geotechnical Engineering
           - Soil mechanics
           - Foundation design
           - Earth retention systems
           - Ground improvement techniques

        Responsibilities:
        1. Provide detailed technical analysis based on engineering principles
        2. Reference relevant building codes and standards (IS, ASTM, BS, etc.)
        3. Suggest innovative and sustainable solutions
        4. Explain complex concepts clearly
        5. Consider cost-effectiveness and practicality
        6. Highlight safety considerations and compliance requirements

        Guidelines:
        1. Base answers on provided context and engineering principles
        2. Include quantitative analysis when applicable
        3. Cite relevant codes and standards
        4. Consider environmental impact
        5. Provide practical, implementable solutions
        6. If information is insufficient, specify what additional data is needed

        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

    def create(self):
        """Create QA chain with custom prompt."""
        model = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            client=genai,
            temperature=self.config.temperature,
        )
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

class StreamlitUI:
    def __init__(self):
        self.config = Config()
        self.vector_store = VectorStore(self.config)
        self.chain = CivilEngineerChain(self.config)

    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AI Civil Engineer Assistant",
            page_icon="🏗️",
            layout="wide"
        )

    def setup_sidebar(self):
        """Configure sidebar elements."""
        with st.sidebar:
            st.title("🏗️ Civil Engineer Assistant")
            st.markdown("---")
            pdf_docs = st.file_uploader(
                "Upload Project Documents (PDF)",
                accept_multiple_files=True,
                help="Upload relevant engineering documents, plans, or specifications"
            )
            
            process = st.button("Process Documents")
            if process and pdf_docs:
                with st.spinner("Processing documents..."):
                    self.process_documents(pdf_docs)
            
            st.markdown("---")
            st.button('Clear Chat History', on_click=self.clear_chat_history)
            
            # Add expertise areas
            st.markdown("### Areas of Expertise")
            st.markdown("""
            - Structural Analysis
            - Construction Management
            - Transportation Engineering
            - Environmental Engineering
            - Geotechnical Engineering
            """)

    def process_documents(self, pdf_docs):
        """Process uploaded PDF documents."""
        try:
            raw_text = PDFProcessor.extract_text(pdf_docs)
            text_chunks = PDFProcessor.create_chunks(raw_text, self.config)
            self.vector_store.create_and_save(text_chunks)
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    def clear_chat_history(self):
        """Clear chat history."""
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Civil Engineering Assistant. Upload project documents and ask me any engineering-related questions."}
        ]

    def process_user_input(self, user_question: str) -> Dict:
        """Process user input and generate response."""
        try:
            db = self.vector_store.load()
            docs = db.similarity_search(user_question)
            chain = self.chain.create()
            return chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return {"output_text": "I apologize, but I encountered an error processing your question. Please try again."}

    def run(self):
        """Run the Streamlit application."""
        self.setup_page()
        self.setup_sidebar()

        # Initialize chat history
        if "messages" not in st.session_state:
            self.clear_chat_history()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Ask your engineering question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = self.process_user_input(prompt)
                    if response:
                        st.markdown(response["output_text"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response["output_text"]}
                        )

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        return
    
    genai.configure(api_key=api_key)
    app = StreamlitUI()
    app.run()

if __name__ == "__main__":
    main()
