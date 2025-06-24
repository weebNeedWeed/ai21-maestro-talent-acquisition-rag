import os
from dotenv import load_dotenv

from pypdf import PdfReader

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NUMBER_OF_CVS = 30
INDEX_NAME = os.getenv("INDEX_NAME") or "ai21-rag"

def prepare_rag_index():
    from pinecone import Pinecone, ServerlessSpec
    import time

    pc = Pinecone(PINECONE_API_KEY)

    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    if INDEX_NAME in pc.list_indexes().names():
        print(f"Index {INDEX_NAME} already exists. Deleting it...\n")
        pc.delete_index(INDEX_NAME)
        time.sleep(10)  # Wait for the index to be deleted
    
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=spec
    )

    # See that it is empty
    print("Index before upsert:")
    print(pc.Index(INDEX_NAME).describe_index_stats())
    print("\n")

def prepare_data():
    cvs_dir = os.path.join(os.path.dirname(__file__), "../ResumesPDF")

    for i in range(1, NUMBER_OF_CVS + 1):
        file_path = cvs_dir + f"/cv ({i}).pdf"
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            reader = PdfReader(file_path)
            
            if len(reader.pages) == 0 :
                print(f"File {file_path} is empty or has no pages.")
                continue
            
            text = ""
            for p in reader.pages:
                text += p.extract_text() + "\n"
                
            text = text.strip()  
            
            handle_text(i, text)
            
            print(f"Extracted text from {file_path}:\n{text}")
            
            
        else:
            print(f"File {file_path} does not exist.")

def handle_text(cv_no, text):
    # print(cv_no, text[:1000])  # Print first 1000 characters of the text for debugging
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_aws import BedrockEmbeddings
    from langchain_pinecone import PineconeVectorStore
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_text(text)
    chunks = text_splitter.create_documents(texts, metadatas=[{"cv_no": cv_no} for _ in range(len(texts))])
    print(f"\nNumber of chunks: {len(chunks)}")
    print("\nExample Chunks:")
    print(chunks)
    
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    vector_store.add_documents(
        documents=chunks,
        ids=[f"cv_{cv_no}_chunk_{i}" for i in range(len(chunks))],
    )
    
    print("After upsert:")
    print(vector_store.index.describe_index_stats())
    print("\n")
    

if __name__ == "__main__" :
    prepare_rag_index()
    prepare_data()
    print("RAG preparation completed.")