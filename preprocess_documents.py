import os
import sys
import random
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def extract_law_metadata(content, filename):
    """
    Extract useful metadata from the law document content.
    
    Args:
        content (str): The content of the law document
        filename (str): The filename of the document
    
    Returns:
        dict: Metadata including law number, year, and title if found
    """
    metadata = {
        "source": filename,
        "law_number": "",
        "year": "",
        "title": ""
    }
    
    # Extract law number and year from filename (e.g., 123_2019.txt -> law_number=123, year=2019)
    file_match = re.match(r'(\d+)_(\d{4})\.txt', os.path.basename(filename))
    if file_match:
        metadata["law_number"] = file_match.group(1)
        metadata["year"] = file_match.group(2)
    
    # Try to extract title from the first 10 lines of content
    lines = content.split('\n')[:10]
    for line in lines:
        # Look for typical law title patterns
        if ("zákon" in line.lower() and len(line) > 15 and len(line) < 200) or \
           ("nariadenie" in line.lower() and len(line) > 15):
            metadata["title"] = line.strip()
            break
    
    return metadata

def process_documents(api_key):
    """
    Process PDF documents from the stiahnute_zakony folder and save the vectorstore to disk.
    
    Args:
        api_key (str): OpenAI API key
    """
    print("Processing PDF files from the stiahnute_zakony folder...")
    
    # Set the correct path to the stiahnute_zakony folder
    stiahnute_zakony_path = "/Users/A200249303/Documents/Zakony/stiahnute_zakony"
    
    # Check if the folder exists
    if not os.path.exists(stiahnute_zakony_path):
        print(f"Error: stiahnute_zakony folder not found at {stiahnute_zakony_path}. Make sure it exists and contains text files.")
        sys.exit(1)
        
    # Load text files from the directory
    loader = DirectoryLoader(
        stiahnute_zakony_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    # Enhance documents with better metadata
    enhanced_docs = []
    for doc in documents:
        try:
            # Extract metadata from content
            enhanced_metadata = extract_law_metadata(doc.page_content, doc.metadata['source'])
            
            # Create a new document with enhanced metadata
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            enhanced_docs.append(enhanced_doc)
        except Exception as e:
            print(f"Error enhancing document {doc.metadata.get('source', 'unknown')}: {e}")
            # Keep the original if enhancement fails
            enhanced_docs.append(doc)
    
    # Shuffle the documents to prevent bias in the vector database
    print("Shuffling documents to improve vector database diversity...")
    random.shuffle(enhanced_docs)
    
    # Split documents into chunks with a larger chunk size for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # Increased from 2000 to 4000 characters
        chunk_overlap=200,  # Increased overlap proportionally
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(enhanced_docs)
    print(f"Documents split into {len(chunks)} chunks.")
    
    # Validate API key format
    if not api_key.startswith("sk-"):
        print("Error: The API key format appears to be invalid. OpenAI API keys typically start with 'sk-'")
        print("Please make sure you're providing a valid API key.")
        sys.exit(1)
        
    try:
        # Create embeddings and vectorstore
        print(f"Creating embeddings with OpenAI API (this may take several minutes)...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Test the API key with a simple embedding
        test_text = "Test API key validity"
        try:
            test_embedding = embeddings.embed_query(test_text)
            print("API key validated successfully!")
        except Exception as e:
            print(f"Error validating API key: {e}")
            sys.exit(1)
            
        print("Creating vector database...")
        # Process chunks in smaller batches to stay within API token limits
        batch_size = 100  # Adjust this number based on your document size
        vectorstore = None
        
        # Shuffle the chunks again to ensure maximum diversity in each batch
        random.shuffle(chunks)
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch = chunks[i:end_idx]
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {i} to {end_idx-1})...")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                batch_vectorstore = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vectorstore)
        
        # Save vectorstore to disk
        save_dir = "./processed_data"
        abs_save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(save_dir)
        
        print(f"Vectorstore successfully saved to {save_dir}")
        print(f"Absolute path: {abs_save_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(save_dir)}")
        print("You can now run the app without needing to reprocess the documents each time.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: OpenAI API key not provided.")
        print("Usage: python preprocess_documents.py YOUR_OPENAI_API_KEY")
        sys.exit(1)
        
    api_key = sys.argv[1]
    process_documents(api_key)
