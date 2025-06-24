# AI21 Maestro Talent Acquisition RAG

A Retrieval-Augmented Generation (RAG) system for talent acquisition that uses AI21's Maestro API to analyze resumes against job descriptions and provide intelligent candidate matching.

## Features

- **Resume Processing**: Automatically extracts text from PDF resumes
- **Vector Search**: Uses Pinecone for similarity-based resume retrieval
- **AI Analysis**: Leverages AI21 Maestro for intelligent candidate evaluation
- **Streamlit UI**: Interactive web interface for job description input and candidate analysis
- **Structured Output**: Provides suitability scores, matching skills, and detailed summaries

## Prerequisites

- Python 3.8+
- AWS Account (for Bedrock embeddings)
- Pinecone Account
- AI21 Account with Maestro API access

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
AI21_API_KEY=your_ai21_api_key
```

### 3. Initialize the RAG System

Run the preparation script to create the Pinecone index and process resumes:

```bash
python src/rag_preparation.py
```

This script will:
- Create a new Pinecone index (or recreate if exists)
- Extract text from all PDF resumes
- Split text into chunks
- Generate embeddings using AWS Bedrock
- Store embeddings in Pinecone

### 4. Launch the UI

Start the Streamlit application:

```bash
streamlit run src/rag_ui.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Enter Job Description**: In the web interface, paste or type the job description you want to match candidates against

2. **Search**: Click the "Search" button to initiate the analysis

3. **Review Results**: The system will:
   - Retrieve the most relevant resume chunks
   - Analyze candidates using AI21 Maestro
   - Return a structured analysis including:
     - Maximum 3 top candidates
     - Suitability scores (1-10)
     - Matching skills and experiences
     - Concise summaries
     - Requirements validation score

## Project Structure

```
ai21-maestro-talent-acquisition-rag/
├── src/
│   ├── rag_preparation.py    # Data processing and index creation
│   └── rag_ui.py            # Streamlit web interface
├── ResumesPDF/              # Directory for PDF resumes
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
└── README.md               # This file
```

## Configuration

### Adjusting Number of Resumes

To process a different number of resumes, modify the `NUMBER_OF_CVS` variable in `rag_preparation.py`:

```python
NUMBER_OF_CVS = 50  # Change to your desired number
```

### Customizing Analysis Requirements

The analysis requirements can be modified in `rag_ui.py` in the `analysis_requirements` list to adjust the AI21 Maestro output format.

### Vector Search Parameters

Adjust the retrieval parameters in `rag_ui.py`:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 7  # Number of chunks to retrieve
    }
)
```