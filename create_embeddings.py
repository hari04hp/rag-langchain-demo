import os
import sys

import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as pc_vector

load_dotenv()

# initialize pinecone
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENV"),
# )

from pinecone import Pinecone, PodSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def create_embeddings(uploaded_files):
    # Read documents
    docs = []
    index_name = "preloaded-index"
    print("file", uploaded_files)
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # First, check if our index already exists. If it doesn't, we create it
    # import pdb
    # pdb.set_trace()
    pc.delete_index(index_name)
    if not pc.list_indexes():
        create_index(index_name, docs, embeddings)
    else:
        for each_index in pc.list_indexes():
            if index_name != each_index['name']:#pinecone.list_indexes():
                create_index(index_name, docs, embeddings)
            else:
                print("Index already exists.")
                break

def create_index(index_name, docs, embeddings):
    # we create a new index
    # pinecone.create_index(name=index_name, metric="cosine", dimension=384)
    pc.create_index(name=index_name, metric="cosine", dimension=384, spec = PodSpec(
    environment="gcp-starter")) #with starter plan
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = pc_vector.from_documents(docs, embeddings, index_name=index_name)
    print("Index Created.")

# Specify the directory containing the PDF files
files_directory = 'files'
file_paths = []

# Check if the directory exists
if os.path.exists(files_directory):
    # List all files in the directory
    for filename in os.listdir(files_directory):
        if filename.endswith(".pdf"):
            file_paths.append(os.path.join(files_directory, filename))
else:
    print(f"Directory '{files_directory}' not found.")
    sys.exit()

# Check if PDF files are found
if not file_paths:
    print("Please add PDF documents to the files folder to continue.")
    sys.exit()

create_embeddings(file_paths)
