import os
import sys
import requests
import re

from tqdm.autonotebook import tqdm, trange

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient

from src.vectordb.create_vectordb import PinconeVectorDb

from semantic_router.encoders import HuggingFaceEncoder


# Global variables
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
EMBEDDING_DIMS = 384
encoder = HuggingFaceEncoder(name=EMBEDDING_MODEL_NAME)

# create Pinecone index instance
pc = PinconeVectorDb()
pc.create_pinecone_index()

# create tool to fetch abstract from arxiv library
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

@tool
def fetch_abstract_from_arxiv(arxiv_id: str) -> str:
    """
    Fetches the abstract of the paper with the specified arXiv ID.
    Args:
        arxiv_id (str): The arXiv ID of the paper.
    Returns:
        str: The abstract of the paper, or an error message if not found.
    """

    res = requests.get(f'https://arxiv.org/abs/{arxiv_id}')
    
    re_match = abstract_pattern.search(res.text)

    return re_match.group(1) if re_match else 'Abstract not found.'


# Web Search tool
@tool('web_search')
def web_search(query: str) -> str:
    """
    Finds general knowledge information using a Tavily search.
    Args:
        query (str): The search query string.
    
    Returns:
        str: A formatted string of the top search results, including title, content, and url.
    """
    print('web search tool called')
    
    search = TavilyClient().search(query=query,
                                    search_depth='advanced',
                                    max_results=5)
    results = search.get('results', [])

    formatted_results = '\n---\n'.join(
        ['\n'.join([x['title'], x['content'], x['url']]) for x in results]
    )
    
    # Return the formatted results or a 'No results found.' message if no results exist.
    return formatted_results if results else 'No results found.'

# RAG search tools

def format_rag_results(matches: list) -> str:
    """
    Formats the RAG search results into a readable format.
    Args:
        matches (list): A list of matches from the RAG search.
    Returns:
        str: A formatted string containing the title, chunk, and arXiv ID of each match.
    """
    formatted_results = []
    
    # Loop through each match and extract its metadata.
    for match in matches:
        text = (
            f"Title: {match['metadata']['title']}\n"
            f"Chunk: {match['metadata']['chunk']}\n"
            f"ArXiv ID: {match['metadata']['arxiv_id']}\n"
        )
        # Append each formatted string to the results list.
        formatted_results.append(text)
    
    # Join all the individual formatted strings into one large string.
    return '\n---\n'.join(formatted_results)

@tool('rag_search_filter')
def rag_search_filter(query: str, arxiv_id: str) -> str:
    '''Finds information from the ArXiv database using a natural language query and a specific ArXiv ID.

    Args:
        query (str): The search query in natural language.
        arxiv_id (str): The ArXiv ID of the specific paper to filter by.
    
    Returns:
        str: A formatted string of relevant document contexts.
    '''
    
    # Encode the query into a vector representation.
    # xq = get_embeddings(query)
    xq = encoder([query])
    
    # Perform a search on the Pinecone index, filtering by ArXiv ID.
    xc = pc.index.query(vector=xq, top_k=6, include_metadata=True, filter={'arxiv_id': arxiv_id})
    
    # Format and return the search results.
    return format_rag_results(xc['matches'])


@tool('rag_search')
def rag_search(query: str) -> str:
    """
    Finds information from the ArXiv database using a natural language query.
    Args:
        query (str): The search query in natural language.
    Returns:
        str: A formatted string of relevant document contexts.
    """
    # Encode the query into a vector representation.
    # xq = get_embeddings(query)
    xq = encoder([query])
    
    # Perform a broader search without filtering by ArXiv ID.
    xc = pc.index.query(vector=xq, top_k=5, include_metadata=True)
    
    # Format and return the search results.
    return format_rag_results(xc['matches'])


# Final Answer Tool
@tool
def final_answer(introduction: str,
                 research_steps: str | list,
                 main_body: str,
                 conclusion: str,
                 sources: str | list) -> str:
    """
    Constructs a final research report from the provided introduction, research steps, main body, conclusion, and sources.
    Args:
        introduction (str): A brief introduction to the research report, introducing the user's question and topic.
        research_steps (str | list): The steps taken in the research process, either as a string or a list of strings.
        main_body (str): The main body of the research report.
        conclusion (str): The conclusion drawn from the research process.
        sources (str | list): The sources used in the research, either as a string or a list of strings.
    Returns:
        str: The final research report as a string.
    """

    # Format research steps if given as a list.
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    
    # Format sources if given as a list.
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    
    # Construct and return the final research report.
    return f'{introduction}\n\n \
        Research Steps:\n{research_steps}\n\n \
            Main Body:\n{main_body}\n\n \
                Conclusion:\n{conclusion}\n\n \
                    Sources:\n{sources}'


