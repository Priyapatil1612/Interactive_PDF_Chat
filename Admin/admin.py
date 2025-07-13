import boto3
import streamlit as st
import os
import uuid


# s3_client
s3_client = boto3.client("s3")
Bucket_Name = os.getenv("Bucket_Name")

# Bedrock
from langchain_community.embeddings import BedrockEmbeddings

# Text Splitter 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# Faais Vector Store
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name = "us-east-1")
bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1",client = bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())

# Split pages into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, splitted_doc):
    vector_store_faiss = FAISS.from_documents(splitted_doc, bedrock_embedding)
    file_name = f"{request_id}.bin"
    folder_path = "/temp/"
    vector_store_faiss.save_local(index_name = file_name, folder_path = folder_path )

    # Upload to S3
    s3_client.upload_file(Filename = folder_path+"/"+file_name+".faiss", Bucket=Bucket_Name, Key = "my_faiss.faiss")
    s3_client.upload_file(Filename = folder_path+"/"+file_name+".pkl", Bucket=Bucket_Name, Key = "my_faiss.pkl")

    return True


# Main Method
def main():
    st.write("Amin side")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Resquest Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode ="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        # Split text in chunks
        splitted_doc = split_text(pages, 1000, 200)
        st.write(f"Splitted Doc Length:{len(splitted_doc)}")
        st.write("===========================")
        st.write(splitted_doc[0])
        st.write("===========================")
        st.write(splitted_doc[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id,splitted_doc)

        if result:
            st.write("PDF processed successfully!")
        else:
            st.write("PDF processing Error! Check Logs..")

if __name__=="__main__":
    main()