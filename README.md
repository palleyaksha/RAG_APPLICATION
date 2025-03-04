This project is live 
https://ragapplication-otmsjo5zqmeyvfee8jgjat.streamlit.app/
# 📄 PDF RAG Chatbot with Gemini AI

## Overview

PDF RAG Chatbot is an interactive Streamlit application that allows users to upload a PDF and chat with its contents using advanced retrieval-augmented generation (RAG) technology. Powered by Google's Gemini AI, this application provides intelligent, context-aware responses based on the uploaded document.

## 🌟 Features

- 📚 PDF Text Extraction
- 🔍 Semantic Search with Vector Embeddings
- 💬 Interactive Chat Interface
- 🤖 AI-Powered Context-Aware Responses
- 📦 Session-Based Vector Database
- 🧹 Easy PDF Clearing

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8+
- pip (Python Package Manager)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🔑 Configuration

1. Create a `.env` file in the project root
2. Add your Gemini API key:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## 📦 Dependencies

- Streamlit
- PyPDF2
- Google GenerativeAI
- python-dotenv
- sentence-transformers
- faiss-cpu
- numpy

## 🚀 Running the Application

```bash
streamlit run app.py
```

## 💡 How to Use

1. Upload a PDF using the sidebar
2. Wait for the PDF to be processed
3. Start asking questions about the document in the chat interface
4. Use the "Clear PDF" button to reset the current document

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/yourusername/pdf-rag-chatbot/issues).

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Google Gemini AI](https://ai.google/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📞 Contact

Palle Yaksha Reddy - palleyaksha28@gmail.com

Project Link: [https://github.com/palleyaksha/pdf-rag-chatbot](https://github.com/palleyaksha/pdf-rag-chatbot)
