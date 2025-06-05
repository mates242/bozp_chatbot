import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def process_documents(api_key):
    """
    Process PDF documents from the stiahnute_zakony folder and save the vectorstore to disk.
    
    Args:
        api_key (str): OpenAI API key
    """
    print("Processing PDF files from the stiahnute_zakony folder...")
    
    # Check if the folder exists
    if not os.path.exists("./stiahnute_zakony/"):
        print("Error: stiahnute_zakony folder not found. Make sure it exists and contains PDF files.")
        sys.exit(1)
        
    # Load PDF files from the directory
    loader = DirectoryLoader(
        "./stiahnute_zakony/", 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
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
