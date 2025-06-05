import os
import streamlit as st
import importlib.util
import sys

# Function to check if a package is available
def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None

# Import basic modules
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Import potentially problematic modules with error handling
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    st.warning(f"Failed to import OpenAIEmbeddings: {e}")
    OpenAIEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.warning(f"Failed to import FAISS: {e}")
    FAISS = None

from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory
# Import OneDrive utilities
from onedrive_utils import download_onedrive_folder

# Set page configuration
st.set_page_config(page_title="Zákony Chat", page_icon="📚", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "using_preprocessed" not in st.session_state:
    st.session_state.using_preprocessed = False

# Function to load preprocessed vectorstore
def load_preprocessed_vectorstore():
    with st.spinner("Načítavam predspracované dokumenty z OneDrive..."):
        # OneDrive public folder link
        onedrive_url = "https://1drv.ms/f/c/6c73abce92e1e313/ElphONm-mWFEsEYR1CqYGBIBv_sKtHeQhVfyiDYKKSu_pQ?e=sqeVOs"
        
        # Local temporary folder to store downloaded files
        temp_data_path = "./temp_onedrive_data"
        
        # Download the vectorstore files from OneDrive
        st.info("Sťahujem vectorstore z OneDrive...")
        download_result = download_onedrive_folder(onedrive_url, temp_data_path)
        
        if download_result:
            st.info(f"Predspracované dáta stiahnuté do: {os.path.abspath(temp_data_path)}")
            try:
                # Load the vectorstore from the downloaded files
                embeddings = OpenAIEmbeddings(
                    api_key=st.session_state.api_key,
                    model="text-embedding-3-small"  # Upgraded embedding model
                )
                
                # Verify that the downloaded files exist and are not empty
                index_faiss_path = os.path.join(temp_data_path, "index.faiss")
                index_pkl_path = os.path.join(temp_data_path, "index.pkl")
                
                if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
                    st.error(f"Missing required files in {temp_data_path}")
                    return False
                
                # Check if index.faiss file is valid
                with open(index_faiss_path, 'rb') as f:
                    header = f.read(4)
                    if header.startswith(b'\x1f\x8b'):  # gzip magic number
                        st.warning("The index.faiss file appears to be in gzip format. Attempting to decompress...")
                        import gzip
                        try:
                            with open(index_faiss_path, 'rb') as f_in:
                                decompressed_content = gzip.decompress(f_in.read())
                            with open(index_faiss_path, 'wb') as f_out:
                                f_out.write(decompressed_content)
                            st.success("Successfully decompressed index.faiss file")
                        except Exception as e:
                            st.error(f"Failed to decompress index.faiss: {e}")
                            return False
                
                # Check if the files exist and are valid before loading
                index_faiss_path = os.path.join(temp_data_path, "index.faiss")
                index_pkl_path = os.path.join(temp_data_path, "index.pkl")
                
                if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
                    st.error(f"Missing required files in {temp_data_path}")
                    st.error("Nepodarilo sa stiahnuť potrebné súbory z OneDrive.")
                    return False
                
                # Try to load the vectorstore
                try:
                    try:
                        vectorstore = FAISS.load_local(temp_data_path, embeddings, allow_dangerous_deserialization=True)
                    except ImportError as imp_error:
                        st.error(f"Failed to import FAISS: {imp_error}")
                        st.error("The FAISS package is not properly installed. Please check your deployment environment.")
                        return False
                    except Exception as e:
                        st.error(f"Failed to load FAISS index: {e}")
                        st.error("Nepodarilo sa načítať dáta stiahnuté z OneDrive.")
                        return False
                
                    # Create conversation chain with improved configuration
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
                    # Add system prompt to improve responses
                    system_prompt = """Si právny expert, ktorý odpovedá na otázky o slovenských zákonoch.
                    Tvoje odpovede by mali byť založené výhradne na obsahu poskytnutých dokumentov a relevantných zákonov.
                    Ak nenájdeš odpoveď na otázku v poskytnutých dokumentoch:
                    1. Jasne uveď, že odpoveď nie je priamo v poskytnutých dokumentoch.
                    2. Neodpovedaj na otázky, pre ktoré nemáš dostatočné informácie.
                    3. Nikdy si nevymýšľaj fakty alebo ustanovenia zákonov.
                    4. Pre otázky o zákonoch, uveď konkrétny zákon a paragraf, ak je táto informácia k dispozícii.
                    
                    Pri poskytovaní odpovedí:
                    - Používaj len informácie z dokumentov, ktoré sú relevantné pre danú otázku.
                    - Ak dokumenty neobsahujú potrebné informácie, povedz: "Prepáčte, ale poskytnuté dokumenty neobsahujú informácie týkajúce sa vašej otázky."
                    - Vždy uvádzaj presný odkaz na príslušný zákon (číslo, názov, paragraf), ak je k dispozícii.
                    """
                    
                    # Create the ChatOpenAI model with system prompt
                    llm = ChatOpenAI(
                        temperature=0, 
                        api_key=st.session_state.api_key, 
                        model_name="gpt-4o",  # Upgraded to GPT-4o
                        model_kwargs={"messages": [{"role": "system", "content": system_prompt}]}
                    )
                    
                    # Create a more selective retriever with higher threshold
                    retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": 5,  # Reduced from 8 to get only the most relevant docs
                            "score_threshold": 0.75  # Increased threshold for better relevance filtering
                        }
                    )
                    
                    conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        chain_type="stuff",
                        return_source_documents=True,
                        verbose=True
                    )
                    
                    st.session_state.conversation = conversation
                    st.session_state.vectorstore = vectorstore
                    st.session_state.docs_processed = True
                    st.session_state.using_preprocessed = True
                    st.success("Predspracované dokumenty úspešne načítané z OneDrive!")
                    return True
                except Exception as e:
                    st.error(f"Chyba pri načítaní predspracovaných dokumentov: {e}")
                    st.error(f"Obsah priečinka: {os.listdir(temp_data_path) if os.path.isdir(temp_data_path) else 'Nie je priečinok'}")
                    return False
            except Exception as e:
                st.error(f"Chyba pri načítaní predspracovaných dokumentov: {e}")
                st.error(f"Obsah priečinka: {os.listdir(temp_data_path) if os.path.isdir(temp_data_path) else 'Nie je priečinok'}")
                return False
        else:
            # Don't try to use local data, only use OneDrive
            st.warning(f"Nepodarilo sa stiahnuť predspracované dokumenty z OneDrive.")
            st.info(f"Aktuálny pracovný priečinok: {os.getcwd()}")
            
            # Retry button for OneDrive download
            if st.button("Skúsiť znova stiahnuť dáta z OneDrive"):
                st.experimental_rerun()
                
            return False

# Function to load and process documents directly
def load_documents():
    # Only try to load from OneDrive, don't fall back to local files
    try:
        if load_preprocessed_vectorstore():
            return
            
        # If OneDrive loading fails, show an error message
        st.error("Nepodarilo sa načítať dokumenty z OneDrive. Aplikácia vyžaduje pripojenie k OneDrive.")
        st.info("Skúste to znova alebo kontaktujte administrátora pre pomoc.")
    except ImportError as e:
        st.error(f"Chyba pri importe knižníc: {e}")
        st.info("Niektoré potrebné knižnice nie sú správne nainštalované. Kontaktujte administrátora.")
    except Exception as e:
        st.error(f"Neočakávaná chyba: {e}")
        st.info("Skúste to znova alebo kontaktujte administrátora pre pomoc.")
    
    st.session_state.docs_processed = False

# Sidebar for API key input
with st.sidebar:
    st.title("📚 Nastavenia")
    
    api_key = st.text_input("Zadajte váš OpenAI API kľúč:", type="password", value=st.session_state.api_key)
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.docs_processed = False
    
    if api_key and not st.session_state.docs_processed:
        st.button("Načítať dokumenty", on_click=load_documents)
        
    if st.session_state.docs_processed:
        if st.session_state.using_preprocessed:
            st.success("✅ Predspracované dokumenty z OneDrive sú pripravené!")
        else:
            st.success("✅ Dokumenty sú pripravené!")
    
# Main area for chat interface
st.title("💬 Zákony AI Chatbot")
st.write("Pýtajte sa na otázky o zákonoch a dostávajte odpovede založené na PDF dokumentoch.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Opýtajte sa niečo o zákonoch..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    if st.session_state.api_key and st.session_state.docs_processed:
        with st.chat_message("assistant"):
            with st.spinner("Premýšľam..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    
                    # For debugging, show the actual query that was sent for embeddings
                    if st.session_state.get("debug_mode", False):
                        st.info(f"Debug - Query: {prompt}")
                    
                    # Write answer first
                    st.write(answer)
                    
                    # Then display source documents if available
                    if "source_documents" in response and len(response["source_documents"]) > 0:
                        # Check if we have relevant documents (based on source)
                        relevant_docs = [doc for doc in response["source_documents"] 
                                        if doc.metadata and "source" in doc.metadata 
                                        and doc.metadata.get("source", "").endswith(".pdf")]
                        
                        if relevant_docs:
                            with st.expander("📄 Zdroje", expanded=False):
                                # Show similarity scores in debug mode
                                if st.session_state.get("debug_mode", False):
                                    st.write("Debug - Document Relevance:")
                                    for i, doc in enumerate(response["source_documents"][:5]):
                                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                                            st.text(f"Doc {i+1} score: {doc.metadata['score']:.4f}")
                                
                                for i, doc in enumerate(relevant_docs[:3]):  # Limit to first 3 relevant sources
                                    source_name = doc.metadata.get("source", "").split("/")[-1]
                                    st.markdown(f"**Zdroj {i+1}:** {source_name}")
                                    # Highlight the key parts of the text if possible
                                    if len(doc.page_content) > 200:
                                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                    else:
                                        st.text(doc.page_content)
                                    st.divider()  # Add divider between sources
                        elif st.session_state.get("debug_mode", False):
                            st.warning("Debug: No relevant sources found with PDF extension")
                except Exception as e:
                    st.error(f"Chyba pri spracovaní otázky: {e}")
                    answer = "Prepáčte, vyskytol sa problém pri spracovaní vašej otázky. Skúste ju preformulovať alebo použite jednoduché ASCII znaky."
                    st.write(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
    elif not st.session_state.api_key:
        with st.chat_message("assistant"):
            st.write("⚠️ Prosím, zadajte najprv váš OpenAI API kľúč v postrannom paneli.")
    elif not st.session_state.docs_processed:
        with st.chat_message("assistant"):
            st.write("⚠️ Prosím, načítajte najprv dokumenty kliknutím na tlačidlo 'Načítať dokumenty' v postrannom paneli.")
