import os
import re

import xml.etree.ElementTree as ET
import requests
import urllib
import feedparser

import pandas as pd
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


OUTPUT_JSON_FILE = 'arxiv_papers.json'
DATA_FOLDER = '../../data'
PDF_FOLDER = '../../data/pdfs/'
OUTPUT_JSON_FILEPATH = os.path.join(DATA_FOLDER, OUTPUT_JSON_FILE)
DF_PDF_CSV_FILE = "arxiv_papers_with_pdfs.csv"
DF_PDF_CSV_FILEPATH = os.path.join(PDF_FOLDER, DF_PDF_CSV_FILE)


def download_arxiv_papers(query: str="natural language processing and large language models", 
                          max_results: int=100, 
                          output_dir: str=DATA_FOLDER, 
                          output_json_file: str=OUTPUT_JSON_FILE) -> pd.DataFrame:
    """
    Downloads the latest papers from the arXiv with the given query and saves them as a JSON file.
    
    Args:
        query (str): The search query for the arXiv. Default is "natural language processing and large language models".
        max_results (int): The maximum number of papers to download. Default is 100.
        output_dir (str): The directory where the JSON file will be saved. Default is DATA_FOLDER.
        output_json_file (str): The name of the JSON file to save. Default is OUTPUT_JSON_FILE.
    Returns:
        pd.DataFrame: A DataFrame containing the downloaded papers.
    """
    try:
        # URL encode the query parameter to replace spaces and special characters
        encoded_query = urllib.parse.quote(query)
        feed_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        feed = feedparser.parse(feed_url)
        
        # Initialize list to store the paper data
        papers = []
        
        # Loop through each entry in the feed
        for entry in feed.entries:
            # Safely extract data with .get() method and set defaults if missing
            title = entry.get("title", "No title available")
            # sanitized_title = sanitize_filename(title) 
            summary = entry.get("summary", "No summary available")
            authors = [author.name for author in entry.get("authors", [])]  # List comprehension with default empty list
            url = entry.get("link", "No URL available")
            pdf_link = next((link.href for link in entry.get("links", []) if link.get("title") == "pdf"), "No PDF link available")
            published = entry.get("published", "No publication date available")
            arxiv_id = entry.get("id", "").split('/')[-1] if "id" in entry else "No arXiv ID available"
            
            # Store each paper's details in a dictionary
            paper = {
                "title": title,
                "summary": summary,
                "authors": authors,
                "url": url,
                "pdf_link": pdf_link,
                "published": published,
                "arxiv_id": arxiv_id
            }
            papers.append(paper)
        
        # Convert to a DataFrame
        df = pd.DataFrame(papers)
        
        # check if the output_json_file exists or not, also the directoies in the path
        if not os.path.exists(output_json_file):
            os.makedirs(output_dir, exist_ok=True)
            output_json_file = os.path.join(output_dir, output_json_file)

        # Save to JSON
        with open(output_json_file, "w", encoding='utf-8') as json_file:
            json.dump(papers, json_file, ensure_ascii=False, indent=4)
        
        print(f"Data saved to {output_json_file}")
    
        return df
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_pdfs(df, download_dir=PDF_FOLDER) -> pd.DataFrame:
    """
    Downloads the PDFs from the provided DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the paper data.
        download_dir (str): The directory where the PDFs will be saved. Default is PDF_FOLDER.
    Returns:
        pd.DataFrame: The DataFrame with the downloaded PDFs added as a new column.
    """
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    pdf_file_names = []  # List to store PDF file names for each paper

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        pdf_url = row["pdf_link"]
        # paper_title = row["title"].replace(" ", "_").replace("/", "-")  # Clean title for filename
        # paper_title = sanitize_filename(row["title"]) # Clean title for filename

        # Define the PDF file path based on the title and download directory
        # pdf_file_path = os.path.join(download_dir, f"{paper_title}.pdf")
        # name = (pdf_url.split('/')[-1]).replace('.', '_')
        
        pdf_file_path = os.path.join(download_dir, pdf_url.split('/')[-1]) + '.pdf'
        # print(pdf_url)
        # print(name)
        # print(pdf_file_path)
        
        try:
            # Check if the PDF already exists to avoid redundant downloads
            if os.path.exists(pdf_file_path):
                print(f"PDF already exists for: {row['title']}")
                pdf_file_names.append(os.path.basename(pdf_file_path))
                continue

            # Download the PDF content
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()  # Check for HTTP request errors

            # Write the content to a file
            with open(pdf_file_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # Filter out keep-alive new chunks
                        pdf_file.write(chunk)
            print(f"Downloaded PDF for: {row['title']}")
            pdf_file_names.append(os.path.basename(pdf_file_path))

        except requests.exceptions.RequestException as e:
            print(f"Failed to download PDF for {row['title']}: {e}")
            pdf_file_names.append("Download failed")  # Log failure in the file name list
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            pdf_file_names.append("Download failed")  # Log generic failure
    # Add the PDF file names to the DataFrame
    df["pdf_file_name"] = pdf_file_names

    # Save the updated DataFrame (optional, if needed for further use)
    df.to_csv(os.path.join(download_dir, DF_PDF_CSV_FILE), index=False)
    
    return df

def download_data():
    # Download the papers and return the DataFrame
    df = download_arxiv_papers(query="natural language processing and large language models", max_results=100)
    df_with_pdfs = download_pdfs(df)
    return df_with_pdfs

if __name__ == "__main__":
    df_with_pdfs = download_data()
