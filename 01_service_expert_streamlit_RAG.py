import streamlit as st 
import os 
import yaml 

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from marketing_analytics_team.agents.product_expert import make_product_expert_agent

# Other Libraries
import pandas as pd
import joblib
import re

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page config
st.set_page_config(
    page_title="Sweet James Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# Configuration constants
MODEL = 'gpt-4o-mini'
EMBEDDING = 'text-embedding-ada-002'
PATH_VECTORDB = "data/data-rag-product-information/services_clean_2.db"

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Choose between Agent or Direct RAG
    mode = st.radio(
        "Select Mode",
        ["🤖 LangGraph Agent", "⚡ Direct RAG Chain"],
        help="Choose between the structured agent or direct RAG pipeline"
    )
    
    # Model settings
    model_name = st.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    # API Key handling
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        try:
            with open('credentials.yml', 'r') as file:
                credentials = yaml.safe_load(file)
                os.environ["OPENAI_API_KEY"] = credentials['openai']
        except:
            st.warning("Please provide an OpenAI API key")
    
    # Initialize based on mode
    if st.button("🚀 Initialize System", type="primary"):
        if "OPENAI_API_KEY" in os.environ:
            with st.spinner(f"Loading {mode}..."):
                try:
                    if mode == "🤖 LangGraph Agent":
                        # Initialize the agent
                        st.session_state.agent = make_product_expert_agent(
                            model=model_name,
                            model_embedding=EMBEDDING,
                            db_path=PATH_VECTORDB
                        )
                        st.session_state.rag_chain = None
                        st.success("✅ Agent initialized successfully!")
                        
                    else:  # Direct RAG Chain
                        # Initialize the direct RAG chain
                        embedding_function = OpenAIEmbeddings(model=EMBEDDING)
                        
                        vectorstore = Chroma(
                            embedding_function=embedding_function, 
                            persist_directory=PATH_VECTORDB
                        )
                        
                        retriever = vectorstore.as_retriever()
                        
                        template = """I'm a Sweet James legal consultant, and I'm here to help you find the right attorney and services for your situation.

Let me share what I know about how Sweet James can help you, based on our current practice areas and expertise.

Sweet James services & expertise:
{context}

What you're asking: {question}

Here's how Sweet James can help:"""
                        
                        prompt = ChatPromptTemplate.from_template(template)
                        
                        model = ChatOpenAI(
                            model=model_name,
                            temperature=0.7,
                        )
                        
                        st.session_state.rag_chain = (
                            {"context": retriever, "question": RunnablePassthrough()}
                            | prompt
                            | model
                            | StrOutputParser()
                        )
                        
                        st.session_state.agent = None
                        st.success("✅ RAG Chain initialized successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error initializing: {str(e)}")
        else:
            st.error("Please provide an OpenAI API key first")

# Main page layout
st.title("⚖️ Sweet James Legal Assistant")
st.markdown("Get expert guidance on personal injury legal services")

# Create columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input handling
    if prompt := st.chat_input("Ask about Sweet James legal services..."):
        if not st.session_state.agent and not st.session_state.rag_chain:
            st.error("Please initialize the system first using the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response based on mode
            with st.spinner("Getting response..."):
                try:
                    if st.session_state.agent:  # Agent mode
                        messages = [HumanMessage(prompt)]
                        result = st.session_state.agent.invoke({"messages": messages})
                        response = result['response'][0].content
                        
                    elif st.session_state.rag_chain:  # Direct RAG mode
                        response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

with col2:
    st.subheader("📊 Chat Statistics")
    
    total_messages = len(st.session_state.messages)
    user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
    
    st.metric("Total Messages", total_messages)
    st.metric("Your Questions", user_messages)
    st.metric("Responses", assistant_messages)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Export chat button
    if st.session_state.messages:
        chat_export = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        
        st.download_button(
            "📄 Export Chat",
            chat_export,
            "sweet_james_chat.txt",
            "text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏛️ Sweet James Legal Assistant | Powered by AI | For informational purposes only</p>
    <p><small>This is not legal advice. Please consult with a Sweet James attorney for specific legal guidance.</small></p>
</div>
""", unsafe_allow_html=True)