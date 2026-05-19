# RAG (Retrieval-Augmented Generation) System

A comprehensive Retrieval-Augmented Generation system built with LangChain, Chroma, and MistralAI. This project provides intelligent document processing and Q&A capabilities with multiple retrieval strategies and LLM providers.

## Features

- 📄 **Multi-format Document Loading**: Support for PDFs, web pages, and text notes
- 🔍 **Advanced Retrieval Strategies**:
  - MMR (Maximal Marginal Relevance) for diverse results
  - ArXiv paper retrieval
  - Multi-retrieval ensemble methods
- 🧠 **Multiple LLM Providers**:
  - MistralAI
  - Groq
  - Google GenAI
  - OpenAI
- 💾 **Vector Storage**: Persistent Chroma vector databases
- 🎨 **Modern Web Interface**: Streamlit-based UI with custom styling
- ⚡ **Fast Embeddings**: Using MistralAI embeddings for semantic search

## Project Structure

```
RAG/
├── app.py                    # Streamlit web application (PDF Intelligence UI)
├── main.py                   # Core RAG logic and prompt templates
├── databse.py               # Database initialization and management
├── requirements.txt         # Python dependencies
├── chroma-db1/              # Vector database storage (Instance 1)
├── chroma-db2/              # Vector database storage (Instance 2)
├── document-loaders/        # Document processing modules
│   ├── pdf.py              # PDF document loader
│   ├── page.py             # Web page loader
│   ├── notes.txt           # Sample notes
│   └── test.py             # Testing utilities
├── retrievers/              # Retrieval strategy implementations
│   ├── my_arxiv.py         # ArXiv paper retriever
│   ├── my_mmr.py           # MMR retriever
│   └── my_mutli.py         # Multi-retrieval ensemble
└── vectorstore/             # Vector store utilities
    └── db.py               # Database operations
```

## Prerequisites

- Python 3.8+
- Virtual environment (venv or conda)
- API keys for:
  - MistralAI (for embeddings and LLM)
  - Groq (optional)
  - Google GenAI (optional)
  - OpenAI (optional)

## Installation

### 1. Clone and Setup

```bash
cd RAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Option 1: Streamlit Web Application (Recommended)

```bash
streamlit run app.py
```

This launches the "Eternal · PDF Intelligence" web interface where you can:
- Upload PDF documents
- Ask questions about the documents
- View intelligent responses powered by RAG

### Option 2: Command Line Interface

```bash
python main.py
```

This starts an interactive RAG session where you can:
- Input questions
- Receive context-aware answers from loaded documents
- Type `0` to exit

## Key Technologies

| Technology | Purpose |
|-----------|---------|
| **LangChain** | LLM orchestration and RAG pipeline |
| **Chroma** | Vector database for embeddings |
| **MistralAI** | Embeddings and LLM inference |
| **Streamlit** | Web UI framework |
| **FastAPI/Uvicorn** | API server support |
| **PyPDF** | PDF document processing |
| **BeautifulSoup** | Web page scraping |

## Module Descriptions

### Document Loaders (`document-loaders/`)
- **pdf.py**: Loads and processes PDF documents using PyPDFLoader
- **page.py**: Extracts and processes web page content
- **test.py**: Testing framework for document loading

### Retrievers (`retrievers/`)
- **my_mmr.py**: Implements Maximal Marginal Relevance for diverse results
- **my_arxiv.py**: Retrieves relevant ArXiv papers
- **my_mutli.py**: Combines multiple retrieval strategies

### Vector Store (`vectorstore/`)
- **db.py**: Manages vector database operations, persistence, and querying

## How RAG Works

1. **Document Ingestion**: Documents are loaded and split into chunks
2. **Embedding**: Text chunks are converted to semantic embeddings using MistralAI
3. **Storage**: Embeddings are stored in Chroma vector database
4. **Query**: User questions are embedded and searched against the database
5. **Retrieval**: Top-k similar documents are retrieved using MMR
6. **Generation**: Retrieved context is combined with LLM prompt to generate answers

## Configuration

### Retrieval Settings (in `main.py`)

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,              # Number of results to return
        "fetch_k": 10,       # Fetch more candidates before MMR
        "lambda_mult": 0.5   # Diversity parameter (0-1)
    }
)
```

### System Prompt Customization

Edit the prompt template in `main.py` to customize AI behavior.

## API Endpoints (FastAPI)

The project includes FastAPI/Uvicorn support for REST API deployment:

```bash
uvicorn app:app --reload
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Ensure all dependencies are installed: `pip install -r requirements.txt` |
| API key errors | Check `.env` file has correct keys and is in project root |
| Slow responses | Reduce `fetch_k` parameter or use fewer documents |
| Memory issues | Limit document batch size in document loaders |

## Performance Tips

- **Reduce K**: Lower `k` value for faster retrieval
- **Text Splitting**: Adjust chunk size in document splitters
- **Embeddings**: Consider using smaller embedding models for speed
- **Caching**: Implement result caching for frequent queries

## Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, HTML)
- [ ] Query optimization and caching layer
- [ ] Fine-tuned embedding models
- [ ] Hybrid search (BM25 + semantic)
- [ ] Multi-language support
- [ ] User authentication for web UI
- [ ] Analytics and logging dashboard

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For issues or questions, please refer to the documentation or create an issue in the repository.
