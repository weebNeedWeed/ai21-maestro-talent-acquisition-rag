import os
from dotenv import load_dotenv

from pypdf import PdfReader

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NUMBER_OF_CVS = 2
INDEX_NAME = "ai21-rag"

def prepare_rag_index():
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(PINECONE_API_KEY)

    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)

    if INDEX_NAME not in pc.list_indexes().names():
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
            
            handle_text(text)
            
            print(f"Extracted text from {file_path}:\n{text}")
            
            
        else:
            print(f"File {file_path} does not exist.")

def handle_text(text):
    pass

if __name__ == "__main__" :
    prepare_rag_index()
    prepare_data()
    print("RAG preparation completed.")