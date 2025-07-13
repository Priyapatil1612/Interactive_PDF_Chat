import boto3
import streamlit as st
import os
import uuid


# s3_client
s3_client = boto3.client("s3")
Bucket_Name = os.getenv("Bucket_Name")

# Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

##
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Text Splitter 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# Faais Vector Store
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name = "us-east-1")
bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1",client = bedrock_client)

folder_path = "/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

def load_index():
    s3_client.download_file(Bucket=Bucket_Name, Key = "my_faiss.faiss", Filename =f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=Bucket_Name, Key = "my_faiss.pkl", Filename =f"{folder_path}my_faiss.pkl")

# def get_llm():
#     llm = Bedrock(model_id = "anthropic.claude-v2:1", client = bedrock_client,
#                   model_kwargs={"max_tokens_to_sample":1024})
#     return llm


def get_llm():
    llm = Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",  # Using instruction-tuned model
        client=bedrock_client,
        model_kwargs={"max_gen_len": 2048, "temperature": 0.2}  
    )
    return llm

def get_response(llm, vectorstore, question):
    # prompt_template = """
    # Human: Please use the given context to provide concise answer to the question
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # <context>
    # {context}
    # </context>

    # Question: {question}

    # Assistant:"""
    prompt_template = """
    You are an AI assistant helping a user find answers from a document. 
    Follow these rules:
    1. Use only the provided context.
    2. If the answer is unclear, say "I don't know."
    3. Be concise but accurate.

    <context>
    {context}
    </context>

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template = prompt_template, input_variables=["context","question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(
            search_type= "similarity", search_kwargs={"k": 10}
        ),
        
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer = qa({"query":question})
    return answer["result"]



# Main Method
def main():
    st.write("User Side to chat with PDF")
    
    load_index()
    dir_list = os.listdir(folder_path)
    st.write(f"Files an Directiories in {folder_path}")
    st.write(dir_list)

    ## Create index
    faiss_index = FAISS.load_local(
        index_name = "my_faiss",
        folder_path = folder_path,
        embeddings = bedrock_embedding,
        allow_dangerous_deserialization = True
    )

    st.write("Index is ready!")

    question = st.text_input("Please ask your Question...")

    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm()
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")


if __name__=="__main__":
    main()