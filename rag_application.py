import os
import streamlit as st
import traceback
from dotenv import load_dotenv
import google.generativeai as genai

# PDF Processing Dependencies
from PyPDF2 import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class PDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_embeddings(self, chunks):
        """Create embeddings for text chunks."""
        return self.model.encode(chunks)
    
    def create_faiss_index(self, embeddings):
        """Create FAISS index for vector search."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

class SessionVectorDatabase:
    @staticmethod
    def save_vector_db_to_session(index, chunks):
        """Save FAISS index and text chunks to Streamlit session state."""
        st.session_state['vector_index'] = index
        st.session_state['vector_chunks'] = chunks
    
    @staticmethod
    def load_vector_db_from_session():
        """Load FAISS index and text chunks from Streamlit session state."""
        if 'vector_index' not in st.session_state or 'vector_chunks' not in st.session_state:
            raise FileNotFoundError("Vector database not found in session. Please upload a PDF first.")
        
        return st.session_state['vector_index'], st.session_state['vector_chunks']
    
    @staticmethod
    def clear_vector_db_from_session():
        """Clear vector database from session state."""
        if 'vector_index' in st.session_state:
            del st.session_state['vector_index']
        if 'vector_chunks' in st.session_state:
            del st.session_state['vector_chunks']
    
    @staticmethod
    def search_vector_db(index, chunks, query_embedding, top_k=3):
        """Perform similarity search in vector database."""
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        
        retrieved_chunks = [chunks[i] for i in indices[0]]
        return retrieved_chunks

class RAGApplication:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.setup_page()
    
    def setup_page(self):
        st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
        st.title("🤖 PDF Retrieval Augmented Generation")
    
    def generate_response(self, query, context):
        """Generate response using Gemini with context."""
        try:
            # Construct a detailed prompt
            full_prompt = f"""
            Context from uploaded PDF:
            {context}

            User Query: {query}

            Please provide a comprehensive and precise answer based on the given context. 
            If the context does not contain sufficient information to answer the query, 
            clearly state that and provide a general response or suggest where more information might be found.
            """
            
            # Use Gemini Pro for text generation
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(full_prompt)
            return response.text
        
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    def run(self):
        # Sidebar for PDF upload
        with st.sidebar:
            st.header("📄 Upload PDF")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            # Add a clear button to remove the current PDF
            if st.button("Clear PDF"):
                SessionVectorDatabase.clear_vector_db_from_session()
                st.session_state.messages = []
                st.experimental_rerun()
            
            if uploaded_file is not None:
                try:
                    # Process PDF
                    text = self.pdf_processor.extract_text_from_pdf(uploaded_file)
                    chunks = self.pdf_processor.chunk_text(text)
                    embeddings = self.pdf_processor.create_embeddings(chunks)
                    index = self.pdf_processor.create_faiss_index(embeddings)
                    
                    # Save vector database to session state
                    SessionVectorDatabase.save_vector_db_to_session(index, chunks)
                    st.success("PDF processed and indexed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    traceback.print_exc()
        
        # Chat interface
        st.subheader("💬 Chat with your PDF")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # Load vector database from session state
                index, chunks = SessionVectorDatabase.load_vector_db_from_session()
                
                # Create query embedding
                query_embedding = self.pdf_processor.model.encode([prompt])
                
                # Retrieve context
                retrieved_context = SessionVectorDatabase.search_vector_db(index, chunks, query_embedding)
                context = " ".join(retrieved_context)
                
                # Generate response
                response = self.generate_response(prompt, context)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except FileNotFoundError:
                st.warning("Please upload a PDF first using the sidebar.")
            except Exception as e:
                st.error(f"Error processing query: {e}")
                traceback.print_exc()

def main():
    app = RAGApplication()
    app.run()

if __name__ == "__main__":
    main()