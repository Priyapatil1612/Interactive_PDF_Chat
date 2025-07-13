# 📄 Interactive PDF Assistant (Powered with AWS)

This project enables users to upload and chat with PDFs using a combination of:

- 💬 **LangChain**
- 🤖 **Amazon Bedrock** (LLMs + Embeddings)
- 📦 **FAISS** for vector search
- 🛢️ **Amazon S3** for index storage
- 🐳 Dockerized for easy deployment
- 🚀 Streamlit as the frontend interface

---

## 🛠️ Features

### ✅ Admin Interface:
- Uploads a PDF  
- Splits it into chunks  
- Generates embeddings using Amazon Titan  
- Creates a FAISS index  
- Uploads the index to S3  

### 🐳 Run the Admin App

# Navigate to the ADMIN folder
cd ADMIN

# Build the Docker image
docker build -t pdf-admin .

# Run the container (replace ../.env with your env path if different)
docker run -p 8083:8083 --env-file ../.env pdf-admin


### ✅ User Interface:
- Downloads the FAISS index from S3  
- Uses Amazon Bedrock LLMs (Claude or LLaMA3)  
- Answers user queries based on PDF content  

### 🐳 Run the User App

# Navigate to the USER folder
cd USER

# Build the Docker image
docker build -t pdf-user .

# Run the container
docker run -p 8084:8083 --env-file ../.env pdf-user

## Folder Structure

interactive-pdf-assistant/
├── ADMIN/
│   ├── admin.py           # PDF upload, chunking, and vector store creation
│   ├── Dockerfile         # Admin Docker setup
│   └── requirements.txt   # Admin dependencies
│
├── USER/
│   ├── app.py             # Chat interface using Bedrock and FAISS
│   ├── Dockerfile         # User Docker setup
│   └── requirements.txt   # User dependencies
│
├── README.md              # Project documentation
├── .gitignore             # Files/folders to exclude from Git
└── .env.example           # Environment variable template