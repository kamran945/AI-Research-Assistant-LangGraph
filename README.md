# Research-Assistant-LangGraph
A research assistant tool powered by LangGraph, inspired by hands-on learning from [LangGraph Mastery: Develop LLM Agents with LangGraph on Udemy] course. https://www.udemy.com/course/langgraph-mastery-develop-llm-agents-with-langgraph
-   This project creates an efficient, responsive, and intelligent research tool that harnesses the power of both proprietary and open-source libraries to deliver a high-quality knowledge retrieval experience.
-   **LangGraph** has been used to build agentic research assistant app.
-   This project is a powerful Research Assistant and Knowledge Retrieval System designed to facilitate deep research and answer complex queries across various fields. 
-   By leveraging a combination of **Hugging Face Hub embeddings**, **Pinecone** as a vector database, **GROQ chat model** as the LLM, **Tavily for web search**, and **retrieval-augmented generation (RAG)** techniques, the system allows for nuanced search capabilities. 
-   Users can extract information from an expansive knowledge base and specialized sources such as ArXiv for academic papers, making it ideal for researchers and knowledge workers.
-   **Streamlit** has been used for graphical user interface.

## How to use:
-   Initialize poetry with `poetry init -n`
-   Run `poetry config virtualenvs.in-project true` so that virtualenv will be present in project directory
-   Run `poetry env use [C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe]` to create virtualenv in project
-   Run `poetry shell`
-   Run `poetry install` to install requried packages
-   Run `streamlit run app.py`

### Workflow:
-   Create template.py and run it `python template.py`
-   Initialize poetry with `poetry init -n`
-   Run `poetry config virtualenvs.in-project true` so that virtualenv will be present in project directory
-   Run `poetry env use [C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe]` to create virtualenv in project
-   Run `poetry shell`
-   Create .env file and enter the following keys: `HUGGINGFACEHUB_API_TOKEN`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `GROQ_API_KEY`, `TAVILY_API_KEY`
-   Create Vector Database in Pinecone
-   Create Tools
-   Create Agents and Graph using LangGraph
-   Create frontend user interface using Streamlit

