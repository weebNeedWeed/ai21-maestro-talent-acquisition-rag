import streamlit as st
import os

from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv
from ai21 import AI21Client
from pypdf import PdfReader
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME") or "ai21-rag"

client = AI21Client(api_key=AI21_API_KEY)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")

vector_store = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 7
    }
)

analysis_requirements = [
    {
        "name": "maximum_of_3_candidates",
        "description": "Ensure that the analysis only considers a maximum of 3 candidates. If more than 3 candidates are found, prioritize the top 3 based on relevance to the job description."
    },
    {
        "name": "extract_candidate_source",
        "description": "Extract the source filename of the candidate's resumé (e.g., 'ResumesPDF/cv ({cv_no}).pdf'). If multiple sources are present, identify the primary one."
    },
    {
        "name": "provide_suitability_score",
        "description": "Provide a suitability score for the candidate on a scale of 1 to 10, where 10 is a perfect match for the job description. The output must be an integer."
    },
    {
        "name": "list_matching_skills",
        "description": "List at least 3, but no more than 7, key skills from the resumé that directly match the requirements in the job description."
    },
    {
        "name": "list_matching_things",
        "description": "List at least 3, but no more than 7, key experiences qualifications, achievements, or other relevant things from the resumé that directly match the requirements in the job description."
    },
    {
        "name": "write_summary",
        "description": "Write a concise summary (40-60 words) explaining the reasoning for the suitability score."
    },
    {
        "name": "check_word_count",
        "description": "The entire response must not exceed 250 words."
    }
]

st.title("AI21 Maestros RAG UI")

with st.form("rag_form"):
    job_desc = st.text_area("Enter job desc:")
    submitted = st.form_submit_button(label="Search")
    
    if submitted:
        if not job_desc:
            st.error("Please enter a job description.")
        else:
            loader = st.success(f"Searching for: {job_desc}")
            
            retrieval_results = retriever.invoke(job_desc)
            
            resume_context = ""
            
            for i, result in enumerate(retrieval_results):
                cv_no = int(result.metadata.get("cv_no"))
                resume_context += f"**CV No {cv_no}**\n"
                path = os.path.join(os.path.dirname(__file__), "../ResumesPDF", f"cv ({cv_no}).pdf")
                if not os.path.exists(path):
                    continue
                reader = PdfReader(path)
                if len(reader.pages) == 0:
                    continue
                for page in reader.pages:
                    resume_context += page.extract_text() + "\n"
                resume_context += "---\n"
                
            st.success("Sending request to AI21 Maestro. This may take a moment")
            
            maestro_input = f"""
                You are an expert HR assistant. Your task is to analyze the provided candidate resumé context and determine if the candidate is a good fit for the given job description.

                **Job Description:**
                {job_desc}

                ---

                **Resumé Context:**
                {resume_context}

                ---

                Based on your analysis, provide a structured summary of the candidate.
                """ 
            
            run = client.beta.maestro.runs.create_and_poll(
                input=maestro_input,
                requirements=analysis_requirements,
                budget="low", # Balanced approach for quality
                include=["requirements_result"], # Ask for the validation report
            )

            st.write(f"Result: {run.result}")
            st.write(f"Requirements Score: {run.requirements_result["score"]}")
            
            
