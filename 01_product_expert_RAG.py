# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a product expert AI agent based on the RAG agent from Clinic #1

# LIBRARIES

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sys
print(sys.version)
print(sys.executable)  # Should show path with langchain_env

# Rag Pipelines 
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Other Libraries
import pandas as pd
import joblib
import re
import os
import yaml

from pprint import pprint
from IPython.display import Markdown

# Backup to display mermaid graphs
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4.1-mini'
EMBEDDING = 'text-embedding-ada-002'
PATH_VECTORDB = "data/data-rag-product-information/services_clean_2.db"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('credentials.yml'))['openai']

# * STEP 1: CREATE THE VECTOR DATABASE

# * Test out loading a single webpage
#   Resource: https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html

# url = "https://university.business-science.io/p/4-course-bundle-machine-learning-and-web-applications-r-track-101-102-201-202a"

url = "https://sweetjames.com/personal-injury/car-accident-lawyers/"

# Create a document loader for the website
loader = WebBaseLoader(url)

# Load the data from the website
documents = loader.load()

print(documents[0].metadata)

dict(documents[0]).keys()

print(documents[0].page_content)


# * Load All Webpages
#   This will take a minute

# df = pd.read_csv("data/data-rag-product-information/products.csv")
df = pd.read_csv("data/data-rag-product-information/services.csv")

df['url']

loader = WebBaseLoader(df['url'].tolist())

documents = loader.load()

documents[1].metadata

len(documents[1].page_content)

joblib.dump(documents, "data/data-rag-product-information/services.pkl")

documents = joblib.load("data/data-rag-product-information/services.pkl")

documents[1].page_content

# * Clean the Beautiful Soup Page Content

def clean_text(text):

    text = re.sub(r'\n+', '\n', text) 
    text = re.sub(r'\s+', ' ', text)  

    text = re.sub(r'Toggle navigation.*?Business Science', '', text, flags=re.DOTALL)
    text = re.sub(r'© Business Science University.*', '', text, flags=re.DOTALL)

    # Replace encoded characters
    text = text.replace('\xa0', ' ')
    text = text.replace('ðŸŽ‰', '')  

    relevant_content = []
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line.lower() for keyword in [
            "car accident", 
            "personal injury", 
            "attorney", 
            "lawyer", 
            "compensation", 
            "settlement", 
            "free consultation", 
            "no fee", 
            "contact us", 
            "call now",
            "experience", 
            "case results", 
            "testimonial",
            "practice areas",
            "motorcycle accident",
            "truck accident", 
            "wrongful death",
            "slip and fall",
            "medical malpractice"
        ]):
            relevant_content.append(line.strip())

    # Join the relevant content back into a single string
    cleaned_text = '\n'.join(relevant_content)

    return cleaned_text

# Test cleaning a single document

pprint(documents[1].page_content)

pprint(clean_text(documents[1].page_content))

pprint(clean_text(documents[0].page_content))


# Clean all documents

documents_clean = documents.copy()

for document in documents_clean:
    document.page_content = clean_text(document.page_content)
    
documents_clean

len(documents_clean)

pprint(documents_clean[1].page_content)

# Assess Length

for document in documents_clean:
    print(document.metadata)
    print(len(document.page_content))
    print("---")


# * Text Embeddings
# OpenAI Embeddings
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

embedding_function = OpenAIEmbeddings(
    model=EMBEDDING,
)

# ** Vector Store - Complete (Large) Documents

# Create the Vector Store (Run 1st Time)
# vectorstore_1 = Chroma.from_documents(
#     documents_clean, 
#     embedding=embedding_function, 
#     persist_directory="data/data-rag-product-information/services_clean_2.db"
# )

# Connect to the Vector Store (Run all other times)
vectorstore_1 = Chroma(
    embedding_function=embedding_function, 
    persist_directory="data/data-rag-product-information/services_clean_2.db"
)

vectorstore_1

vectorstore_1.similarity_search("vehicle accident lawyer", k = 6)

retriever_1 = vectorstore_1.as_retriever()

# * Prompt template 

template = """I'm a Sweet James legal consultant, and I'm here to help you find the right attorney and services for your situation.

Let me share what I know about how Sweet James can help you, based on our current practice areas and expertise.

Sweet James services & expertise:
{context}

What you're asking: {question}

Here's how Sweet James can help:"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification

model = ChatOpenAI(
    model = MODEL,
    temperature = 0.7,
)

response = model.invoke("which service is best for a car accident involving a truck?")

pprint(response.content)

print(response.content)

# * RAG Chain

rag_chain_1 = (
    {"context": retriever_1, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain_1.invoke("which service is best for a car accident involving a truck?")

Markdown(result)

pprint(result)


result = rag_chain_1.invoke("What if i slip and fall at a store? which attorney should i consider at sweet james?")

Markdown(result)

pprint(result)

result = rag_chain_1.invoke("Which services are provided by Sweet James?")

Markdown(result)

pprint(result)

# * STEP 2: MAKE THE RAG AGENT
#  - Create a RAG Agent based on the one used in Clinic #1
#  - Modularize the agent for easier re-use in production
#  - Use LangGraph to manage State
#  - Implement LangGraph Messages History to track multi-agent conversations

# Libraries 
from marketing_analytics_team.agents.product_expert import make_product_expert_agent

# Make the agent

product_expert_agent = make_product_expert_agent(
    model=MODEL,
    model_embedding=EMBEDDING,
    db_path=PATH_VECTORDB
)

product_expert_agent

display(Image(product_expert_agent.get_graph().draw_png()))

# product_expert_agent.get_input_jsonschema()['properties']



# * TEST: What services does Sweet James provide?

messages = [
    HumanMessage("what services does Sweet James provide?")
]

result = product_expert_agent.invoke({"messages": messages})

result.keys()

result['response']

Markdown(result['response'][0].content)

pprint(result['response'][0].content)


# * TEST: What services are best for a car accident involving a truck?

messages = [
    HumanMessage("which sweet james services are best for a car accident involving a truck?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

pprint(result['response'][0].content)

# * TEST: What should I do if I'm involved in a truck accident?

messages = [
    HumanMessage("I was rear-ended by a delivery truck while stopped at a red light. The driver was texting. What Sweet James services can help me and what should I expect?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

# * TEST: What should I do if I slip and fall at a store?

messages = [
    HumanMessage("My elderly mother fell at a grocery store because of a wet floor with no warning signs. She broke her hip and needs surgery. Does Sweet James handle these cases and what's the process?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

# * TEST: How much is a consultation with Sweet James?

messages = [
    HumanMessage("How much is a consultation with Sweet James?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

# * TEST: How does the 'No Fee' policy work?

messages = [
    HumanMessage("I can't afford to pay attorney fees upfront. How does Sweet James' 'No Fee' policy actually work? What costs might I still be responsible for?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)


# * TEST: Should I accept an insurance settlement offer?
messages = [
    HumanMessage("The insurance company offered me $15,000 for my motorcycle accident injuries, but my medical bills are already $12,000. Should I accept this or contact Sweet James first?")
]

result = product_expert_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)

