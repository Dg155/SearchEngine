from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from pymilvus import DataType, MilvusClient
from regex import F
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup
import json
import os
import re


# Variables
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "Alibaba-NLP/gte-large-en-v1.5"
encoder = SentenceTransformer(model_name, device=DEVICE)

# Constants
MAX_SEQ_LENGTH = encoder.get_max_seq_length() 
EMBEDDING_LENGTH = encoder.get_sentence_embedding_dimension()
FOLDER_PATH = r"C:\Users\kidro\Desktop\SearchEngine\ANALYST"
DATABASE_NAME = "ANALYSTLLM"
client = MilvusClient(
    uri="https://in03-2893550dac2ce6c.api.gcp-us-west1.zillizcloud.com",
    token="69982138f98274926e691ee7f5c44a775f816ce0065d51fbb3fa95fbdb6be466c9f2d3c603354518c8cafcce30af0b3948049dcc"
)

# Create sizes for the chunks of texts to embed
chunk_size = MAX_SEQ_LENGTH
#chunk_size = 8192
chunk_overlap = np.round(chunk_size * 0.15, 0)

# Create an instance of the RecursiveCharacterTextSplitter
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = len,)

def ReadJSONFile(filePath):
    with open(filePath, "r") as f:
        jsonData = json.load(f)
        soup = BeautifulSoup(jsonData["content"], "html.parser", from_encoding=jsonData["encoding"])
        title = soup.find('title').get_text() if soup.find('title') else "No Title Found"
        return title, jsonData["url"], soup

def GatherFiles(folderPath):
    file_list = []
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            # Only read json files
            if file.endswith(".json"):
                file_list.append(os.path.join(root, file))
    
    return file_list


def SplitHTML(filePath):
    # Keep track of time for debugging
    start_time = time.time()
    
    html_header_splits = []

    # Open the json file and load it into memory
    title, url, soup = ReadJSONFile(filePath)

    html_header_splits.append({
            'page_content': soup.get_text(),
            'metadata': {
                'source': url,
                'title' : title
            }})

    # Split the documents further into smaller chunks
    chunks = child_splitter.split_documents(html_header_splits)
    print(f"doc: {filePath} split into chunks: {len(chunks)}") 
    end_time = time.time()
    print(f"chunking time: {end_time - start_time}")
    return chunks

def ParseFiles(folderPath):
    file_list = GatherFiles(folderPath)
    chunks = []
    chunks += SplitHTML(file_list[0])
    return chunks
        

def CreateEmbeddings(chunkList):
    embedded_chunks = []
    for chunk in chunkList:
        # Generate embeddings using encoder from HuggingFace.
        embeddings = torch.tensor(encoder.encode([chunk.page_content]))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        converted_values = list(map(np.float32, embeddings))[0]

        chunk_dict = {
            'vector': converted_values,
            'text': chunk.page_content,
            'source': chunk.metadata['source'],
            'title': chunk.metadata['h1'][:50],
        }
        embedded_chunks.append(chunk_dict)

    return embedded_chunks

def UploadEmbeddings(embeddings):
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    #field = ["id", "title", "title_vector", "link", "content", "content_vector"]
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=MAX_SEQ_LENGTH)
    #schema.add_field(field_name="title_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=MAX_SEQ_LENGTH)
    #schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="content_vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_LENGTH)


    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="id"
    )

    #index_params.add_index(
    #    field_name="title_vector", 
    #    index_type="AUTOINDEX",
    #    metric_type="IP"
    #)

    index_params.add_index(
        field_name="content_vector", 
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=DATABASE_NAME,
        schema=schema,
        index_params=index_params
    )

    res = client.insert(
        collection_name= DATABASE_NAME,
        data= embeddings
        )


if __name__ == "__main__":
    combined_chunks = ParseFiles(FOLDER_PATH)
    print("Looking at a sample chunk...")
    #print(combined_chunks[1].page_content[:100])
    #print(combined_chunks[1].metadata)

    embeddingList = CreateEmbeddings(combined_chunks)
    UploadEmbeddings(embeddingList)

    