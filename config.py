import os
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm

# Set up OpenAI API key from environment variable
# Make sure to set: export OPENAI_API_KEY="your-actual-api-key" in your terminal
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please run: export OPENAI_API_KEY='your-api-key'")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize Milvus client, LLM, and embedding model
# Milvus Client Configuration Options:
# 1. Local file (Milvus Lite) - most convenient for development and small datasets
milvus_client = MilvusClient(uri="./milvus.db")

# 2. For large scale data, use Milvus server on Docker/Kubernetes:
# milvus_client = MilvusClient(uri="http://localhost:19530")

# 3. For Zilliz Cloud (fully managed cloud service):
# milvus_client = MilvusClient(
#     uri="YOUR_ZILLIZ_CLOUD_PUBLIC_ENDPOINT", 
#     token="YOUR_ZILLIZ_CLOUD_API_KEY"
# )

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
