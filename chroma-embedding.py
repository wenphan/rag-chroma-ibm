# Ignore SSL just for dev
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Get data 
import os
import wget

filename = "state_of_the_union.txt"
url = "https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"

if not os.path.isfile(filename):
	wget.download(url, out=filename)

# Load data as documents
from langchain.document_loaders import TextLoader
loader = TextLoader(filename)
documents = loader.load()
print("Number of documents: " + str(len(documents)))
print("Example document (first 100 characters): " + str(documents[0].page_content)[:100])

# Split documents
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print("\n")
print("Number of texts: " + str(len(texts)))
print("Example texts: " + str(texts[0].page_content))

# Insert texts into vector store with embeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
chroma_vectorstore = Chroma.from_documents(texts, embeddings)
print("\n")
print(chroma_vectorstore)
