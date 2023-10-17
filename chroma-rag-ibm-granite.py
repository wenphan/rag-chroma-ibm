# Ignore SSL just for dev
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Environment variables and keys
import os
credentials = {
	"url": "https://us-south.ml.cloud.ibm.com",
	"apikey": os.environ["WXA_API_KEY"]
}

wxa_project_id = os.environ["WXA_PROJECT_ID"]

# Query
query = "What did the president say about Ketanji Brown Jackson?"

# IBM Model Completion with LangChain
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
	GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
	GenParams.MIN_NEW_TOKENS: 1,
	GenParams.MAX_NEW_TOKENS: 100
}

model = Model(
	model_id=ModelTypes.GRANITE_13B_INSTRUCT,
	params=parameters,
	credentials=credentials,
	project_id=wxa_project_id
)

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
granite_llm_ibm = WatsonxLLM(model=model)
result = granite_llm_ibm(query)
print("Basic LLM generation")
print("---------------------------------------")
print("Query: " + query)
print("Result:" + result)

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

# Split documents
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Insert texts into vector store with embeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
chroma_vectorstore = Chroma.from_documents(texts, embeddings)

# Similarity search
print("\n")
print("Similarity search")
print("---------------------------------------")
texts_sim = chroma_vectorstore.similarity_search(query, k=3)
print("Number of relevant texts: " + str(len(texts_sim)))

print("\n")
print("First 100 characters of relevant texts.")
for i in range(len(texts_sim)):
	print("Text " + str(i) + ": " + str(texts_sim[i].page_content[0:100]))

# RAG generation with explicit relevant texts
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(granite_llm_ibm, chain_type="stuff")
result = chain.run(input_documents=texts_sim, question=query)
print("\n")
print("RAG generation with explicit relevant texts")
print("---------------------------------------")
print("Query: " + query)
print("Result:" + result)

# RAG QA chain
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=granite_llm_ibm, chain_type="stuff", retriever=chroma_vectorstore.as_retriever())
result = qa.run(query)
print("\n")
print("RAG QA chain")
print("---------------------------------------")
print("Query: " + query)
print("Result:" + result)
