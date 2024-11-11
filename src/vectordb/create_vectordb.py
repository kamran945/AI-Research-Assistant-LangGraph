import os
import time
from getpass import getpass

from tqdm.auto import tqdm

import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, Index

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)

# Global variables
OUTPUT_JSON_FILE = 'arxiv_papers.json'
DATA_FOLDER = '../data'
PDF_FOLDER = '../data/pdfs/'
OUTPUT_JSON_FILEPATH = os.path.join(DATA_FOLDER, OUTPUT_JSON_FILE)
DF_PDF_CSV_FILE = "arxiv_papers_with_pdfs.csv"
DF_PDF_CSV_FILEPATH = os.path.join(PDF_FOLDER, DF_PDF_CSV_FILE)


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

INDEX_NAME = 'langgraph-research-agent'
BATCH_SIZE = 64

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
EMBEDDING_DIMS = 384



def load_and_chunk_pdf(pdf_file_name: str, 
                       saved_dir: str=PDF_FOLDER,
                       chunk_size: int=CHUNK_SIZE, 
                       chunk_overlap: int=CHUNK_OVERLAP) -> list[str]:
    """
    Loads a PDF file into chunks and returns a list of chunks.
    Args:
        pdf_file_name (str): The name of the PDF file.
        saved_dir (str): The directory where the PDF file is saved. Default is PDF_FOLDER.
        chunk_size (int): The size of each chunk in bytes. Default is CHUNK_SIZE.
        chunk_overlap (int): The overlap between chunks in bytes. Default is CHUNK_OVERLAP.
    Returns:
        List[str]: A list of chunks from the PDF file.
    """

    print(f'Loading and splitting into chunks: {pdf_file_name}')
    # name = remove_dot_from_filename(pdf_file_name)
    # print(name)
    
    pdf_file_path = os.path.join(saved_dir, pdf_file_name)

    # Load the PDF file into a DocumentLoader object
    loader = PyPDFLoader(pdf_file_path)
    data = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)

    return chunks


def add_chunks_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds chunks to the DataFrame, including their IDs and metadata.
    Args:
        df (pd.DataFrame): The DataFrame containing the paper details.
    Returns:
        pd.DataFrame: The updated DataFrame with added chunk information.
    """

    expanded_rows = []  # List to store expanded rows with chunk information

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        try:
            chunks = load_and_chunk_pdf(row['pdf_file_name'])
        except Exception as e:
            print(f"Error processing file {row['pdf_file_name']}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            pre_chunk_id = i-1 if i > 0 else ''  # Preceding chunk ID
            post_chunk_id = i+1 if i < len(chunks) - 1 else ''  # Following chunk ID

            expanded_rows.append({
                'id': f"{row['arxiv_id']}#{i}",  # Unique chunk identifier
                'title': row['title'],
                'summary': row['summary'],
                'authors': row['authors'],
                'arxiv_id': row['arxiv_id'],
                'url': row['url'],
                'chunk': chunk.page_content,  # Text content of the chunk
                'pre_chunk_id': '' if i == 0 else f"{row['arxiv_id']}#{pre_chunk_id}",  # Previous chunk ID
                'post_chunk_id': '' if i == len(chunks) - 1 else f"{row['arxiv_id']}#{post_chunk_id}"  # Next chunk ID
            })
    # Return a new expanded DataFrame
    return pd.DataFrame(expanded_rows)




def get_embeddings(model_name, texts):
    # Define the Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    embeddings_list = []
    for text in texts:
        embeddings_list.append(embeddings.embed_query(text))

    return embeddings_list


class PinconeVectorDb:
    def __init__(self, 
                 cloud: str='aws', 
                 region: str='us-east-1'):
        """
        Initialize the PinconeVectorDb class.
        Args:
            cloud (str): The cloud provider for Pinecone. Default is 'aws'.
            region (str): The AWS region for Pinecone. Default is 'us-east-1'.
        """
        # Check if 'PINECONE_API_KEY' is set; prompt if not
        self.pc_api_key = os.getenv('PINECONE_API_KEY') or getpass('Pinecone API key: ')
        self.pc, self.spec = self.initialize_pinecone_client(cloud=cloud, 
                                                             region=region)
    
    def initialize_pinecone_client(self, cloud: str, region: str):
        """
        Initialize the Pinecone client and return it.
        Args:
            cloud (str): The cloud provider for Pinecone. 
            region (str): The AWS region for Pinecone.
        Returns:
            Pinecone client and serverless specification objects.
        """

        # Initialize the Pinecone client
        pc = Pinecone(api_key=self.pc_api_key)
        # Define the serverless specification for Pinecone (AWS region 'us-east-1')
        spec = ServerlessSpec(
            cloud=cloud, 
            region=region
        )

        return pc, spec
    
    def create_pinecone_index(self,
                                index_name: str=INDEX_NAME, 
                                EMBEDDING_DIMS: int=EMBEDDING_DIMS, 
                                metric: str='cosine') -> Index:
        """
        Creates a Pinecone index with the given name and dimensions.
        Args:
            index_name (str): The name of the index. Default is INDEX_NAME.
            EMBEDDING_DIMS (int): The dimensionality of the embeddings. Default is EMBEDDING_DIMS.
            metric (str): The metric used for similarity. Default is 'cosine'.
        Returns:
            Pinecone index object.
        """

        # Check if the index exists; create it if it doesn't
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                index_name,
                dimension=EMBEDDING_DIMS,  # Embedding dimension
                metric=metric,
                spec=self.spec  # Cloud provider and region specification
            )

            # Wait until the index is fully initialized
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        else:
            print(f"Index {index_name} already exists.")

        # Connect to the index
        self.index = self.pc.Index(index_name)

        # Add a short delay before checking the stats
        time.sleep(1)

        # View the index statistics
        print(f"Index Stats:\n{self.index.describe_index_stats()}")
    
    def get_embeddings(self,
                       texts: list[str],
                       model_name: str=EMBEDDING_MODEL_NAME):
        # Define the Hugging Face Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        embeddings_list = []
        for text in texts:
            embeddings_list.append(embeddings.embed_query(text))

        return embeddings_list
    
    def add_embeddings_to_index(self, 
                                data: pd.DataFrame, 
                                batch_size: int=BATCH_SIZE,):

        # data = expanded_df
        # batch_size = 64  # Set batch size

        # Loop through the data in batches, using tqdm for a progress bar
        for i in tqdm(range(0, len(data), batch_size)):
            i_end = min(len(data), i + batch_size)  # Define batch endpoint
            batch = data[i:i_end].to_dict(orient='records')  # Slice data into a batch

            # Extract metadata for each chunk in the batch
            metadata = [{
                'arxiv_id': r['arxiv_id'],
                'title': r['title'],
                'chunk': r['chunk'],
            } for r in batch]
            
            # Generate unique IDs for each chunk
            ids = [r['id'] for r in batch]
            
            # Extract the chunk content
            chunks = [r['chunk'] for r in batch]
            
            # Convert chunks into embeddings
            embeds = self.get_embeddings(chunks)
            
            # Upload embeddings, IDs, and metadata to Pinecone
            self.index.upsert(vectors=zip(ids, embeds, metadata))
            
            # View the index statistics
            print(f"Index Stats:\n{self.index.describe_index_stats()}")


if __name__ == "__main__":
    
    # Load the dataset with PDF files and add chunks to the DataFrame
    df_with_pdfs = pd.read_csv(DF_PDF_CSV_FILEPATH)
    df_with_chunks = add_chunks_to_df(df_with_pdfs)
    
    # Initialize the Pincone VectorDb class and create the index
    pc = PinconeVectorDb()
    pc.create_pinecone_index()
    pc.add_embeddings_to_index(data=df_with_chunks)