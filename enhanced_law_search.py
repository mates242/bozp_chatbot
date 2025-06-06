"""
Enhanced Law Search Module for the chatbot
This module provides improved law search capabilities that ensure searches are specifically directed
at the vectorstore in processed_data rather than directly looking in stiahnute_zakony_html files.
"""

import re
import os
import logging
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Define the path to the stiahnute_zakony directory - adjust this path as needed
LAWS_DIRECTORY = '/Users/A200249303/Documents/Zakony/stiahnute_zakony'

def direct_file_search(law_number, year, debug_mode=False):
    """
    Search directly for a law file in the stiahnute_zakony directory.
    This is a fallback mechanism when vectorstore search fails.
    """
    if debug_mode:
        st.write(f"Vykonávam priame vyhľadávanie súboru pre zákon {law_number}/{year} v adresári stiahnute_zakony")
    
    # Create file name patterns to check
    file_patterns = [
        f"{law_number}_{year}.txt", 
        f"{law_number}-{year}.txt",
        f"{law_number}_{year[2:]}.txt",  # Handle two-digit year format
        f"{law_number}-{year[2:]}.txt"   # Handle two-digit year format with dash
    ]
    
    # Check if directory exists
    if not os.path.exists(LAWS_DIRECTORY):
        if debug_mode:
            st.error(f"Adresár {LAWS_DIRECTORY} nebol nájdený!")
        return None
    
    # List all files in the directory
    try:
        all_files = os.listdir(LAWS_DIRECTORY)
        
        # First check for exact file name matches
        for pattern in file_patterns:
            if pattern in all_files:
                file_path = os.path.join(LAWS_DIRECTORY, pattern)
                if debug_mode:
                    st.success(f"Našiel som súbor zákona: {pattern}")
                
                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    return {"file_path": file_path, "content": content}
                except Exception as e:
                    if debug_mode:
                        st.error(f"Chyba pri čítaní súboru {pattern}: {str(e)}")
        
        # If exact match not found, try partial match
        for file in all_files:
            # Look for patterns like "75_2023" or variations in filenames
            if (f"{law_number}_{year}" in file or 
                f"{law_number}-{year}" in file or 
                f"{law_number}/{year}" in file):
                
                file_path = os.path.join(LAWS_DIRECTORY, file)
                if debug_mode:
                    st.success(f"Našiel som súbor zákona s čiastočnou zhodou: {file}")
                
                # Read the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {"file_path": file_path, "content": content}
                except Exception as e:
                    if debug_mode:
                        st.error(f"Chyba pri čítaní súboru {file}: {str(e)}")
        
        if debug_mode:
            st.warning(f"Nenašiel som súbor pre zákon {law_number}/{year} v adresári {LAWS_DIRECTORY}")
        return None
        
    except Exception as e:
        if debug_mode:
            st.error(f"Chyba pri prehľadávaní adresára {LAWS_DIRECTORY}: {str(e)}")
        return None

def direct_law_search(law_number, year, vectorstore, debug_mode=False):
    """
    Enhanced function for direct law searching in the vectorstore
    Focuses on using the vectorstore in processed_data for all searches
    """
    # Expanded formats to try, with more variations to increase match chance
    formats = [
        f"{law_number}_{year}",
        f"{law_number}/{year}",
        f"{law_number}-{year}",
        f"{law_number}.{year}",
        f"zákon {law_number}_{year}",
        f"zákon {law_number}/{year}",
        f"zákon č. {law_number}/{year}",
        f"{law_number}/{year} Z.z.",
        f"{law_number}_{year} Z.z.",
        f"{law_number}/{year} Zb.",
        f"číslo {law_number} z roku {year}",
        f"č. {law_number}/{year}"
    ]
    
    # Try more advanced search approaches first
    if debug_mode:
        st.write(f"Hľadám zákon {law_number}/{year} alebo {law_number}_{year} vo vektorovom úložisku (processed_data)")
    
    # Try MMR retrieval first for better results
    try:
        # Combined search using both formats for higher chance of finding relevant documents
        combined_query = f"{law_number}/{year} OR {law_number}_{year} OR zákon {law_number} {year}"
        
        if debug_mode:
            st.write(f"Skúšam MMR retrieval s kombinovaným dotazom: {combined_query}")
        
        # MMR retrieval to get more diverse matches
        mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 15})
        mmr_docs = mmr_retriever.get_relevant_documents(combined_query)
        
        if mmr_docs:
            # Check if any document contains our law in content or metadata
            for doc in mmr_docs:
                content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
                source = doc.metadata.get('source', '').lower() if hasattr(doc, 'metadata') else ""
                
                # Check for exact law identifier in various formats
                if any(f"{law_number}{sep}{year}" in content or f"{law_number}{sep}{year}" in source 
                       for sep in ['/', '_', '-', '.']):
                    if debug_mode:
                        st.success(f"Našiel som zákon pomocou MMR retrieval v vektorovom úložisku")
                    return mmr_docs
    except Exception as mmr_error:
        if debug_mode:
            st.error(f"Chyba pri MMR vyhľadávaní: {str(mmr_error)}")
    
    # Try explicit similarity search with various formats
    for format_str in formats:
        if debug_mode:
            st.write(f"Skúšam explicitný formát: {format_str}")
        
        try:
            # Slightly increase k for better coverage
            docs = vectorstore.similarity_search(format_str, k=7)
            
            if docs:
                # Validate results - check if documents actually contain the law reference
                for doc in docs:
                    content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
                    
                    # Check content - expanded checks to catch more variations
                    patterns_to_check = [
                        f"{law_number}_{year}",
                        f"{law_number}/{year}",
                        f"{law_number}-{year}",
                        f"{law_number} {year}",
                        fr"{law_number}\s*/\s*{year}",
                        fr"{law_number}\s*_\s*{year}",
                        fr"zákon.*{law_number}.*{year}"
                    ]
                    
                    for pattern in patterns_to_check:
                        if re.search(pattern, content, re.IGNORECASE):
                            if debug_mode:
                                st.success(f"Našiel som zákon {pattern} v obsahu dokumentu")
                                # Show context around the match
                                match = re.search(pattern, content, re.IGNORECASE)
                                if match:
                                    start_idx = max(0, match.start() - 25)
                                    end_idx = min(len(content), match.end() + 25)
                                    st.write(f"Kontext: ...{content[start_idx:end_idx]}...")
                            return docs
                    
                    # Check metadata as well
                    if hasattr(doc, 'metadata'):
                        source = doc.metadata.get('source', '').lower()
                        if any(pattern in source for pattern in [
                            f"{law_number}_{year}", f"{law_number}/{year}", 
                            f"{law_number}-{year}", f"{law_number}.{year}"
                        ]):
                            if debug_mode:
                                st.success(f"Našiel som zákon v metadátach: {source}")
                            return docs
        except Exception as e:
            if debug_mode:
                st.error(f"Chyba pri priamom vyhľadávaní: {str(e)}")
    
    # Final attempt - broad sweep with general law terms and filtering by exact match
    try:
        # Get a broader set of documents about laws
        broad_query = f"zákony právne predpisy legislatíva {law_number} {year}"
        if debug_mode:
            st.write(f"Posledný pokus - široké vyhľadávanie: {broad_query}")
        
        # Increase k significantly for this final attempt
        raw_docs = vectorstore.similarity_search(broad_query, k=20)
        
        # Filter by exact reference match
        for doc in raw_docs:
            content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
            
            # Look for exact law references
            if any(exact_ref in content for exact_ref in [
                f"{law_number}/{year}", f"{law_number}_{year}", 
                f"{law_number}-{year}", f"{law_number}.{year}"
            ]):
                if debug_mode:
                    st.success(f"Našiel som zákon v finálnom pokuse")
                return [doc]  # Found a matching document
    except Exception as e:
        if debug_mode:
            st.error(f"Chyba pri finálnom vyhľadávaní: {str(e)}")
    
    # No documents found in vectorstore, try direct file search as fallback
    if debug_mode:
        st.warning(f"Nenašiel som žiadne dokumenty vo vektorovom úložisku, skúšam priame vyhľadávanie súborov")
    
    # Try direct file search as fallback
    direct_result = direct_file_search(law_number, year, debug_mode)
    if direct_result:
        # Create a document object similar to what would be returned from vectorstore
        from langchain.schema import Document
        
        doc = Document(
            page_content=direct_result["content"],
            metadata={"source": f"stiahnute_zakony/{os.path.basename(direct_result['file_path'])}"}
        )
        return [doc]
    
    # No documents found anywhere
    if debug_mode:
        st.warning(f"Nenašiel som žiadne dokumenty obsahujúce zákon {law_number}/{year}")
    return None

def get_law_content(law_number, year, vectorstore, chat_history=None, debug_mode=False):
    """
    Comprehensive function to retrieve law content from the vectorstore
    Uses multiple search strategies to find information about a specific law
    """
    if chat_history is None:
        chat_history = []
        
    if debug_mode:
        st.write(f"Hľadám obsah zákona {law_number}/{year} vo vektorovom úložisku")
    
    # First try to get documents using our enhanced direct search
    direct_docs = direct_law_search(law_number, year, vectorstore, debug_mode)
    
    if not direct_docs:
        if debug_mode:
            st.warning(f"Nenašiel som žiadne dokumenty pre zákon {law_number}/{year}")
        return None
        
    # Create an optimized prompt for this specific law
    law_prompt = (
        f"Na základe poskytnutých dokumentov, vysvetli čo presne upravuje zákon {law_number}/{year} "
        f"alebo {law_number}_{year}. O čom je tento zákon, aké má hlavné body a účel? "
        f"Ak sa v dokumentoch nenachádza tento zákon, povedz mi to priamo."
    )
    
    # Use a specialized chain for getting precise information
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    law_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True
    )
    
    # Get the response
    response = law_chain({"question": law_prompt, "chat_history": chat_history})
    
    # Validate that we got a useful response
    if response["source_documents"] and not any(phrase in response["answer"].lower() for phrase in 
        ["neviem", "nemám informácie", "nenašiel som", "nemám dostatok", "nenašiel sa"]):
        
        if debug_mode:
            st.success(f"Našiel som informácie o zákone {law_number}/{year}")
        return response
        
    # If we didn't get a good answer, try with a different prompt format
    alternative_prompt = f"Informácie o zákone {law_number}_{year} alebo {law_number}/{year}"
    if debug_mode:
        st.write(f"Skúšam alternatívny prompt: {alternative_prompt}")
        
    alt_response = law_chain({"question": alternative_prompt, "chat_history": []})
    
    # Check if alternative approach worked
    if alt_response["source_documents"] and not any(phrase in alt_response["answer"].lower() for phrase in 
        ["neviem", "nemám informácie", "nenašiel som", "nemám dostatok", "nenašiel sa"]):
        
        if debug_mode:
            st.success(f"Našiel som informácie o zákone pomocou alternatívneho promptu")
        return alt_response
        
    # No good response found
    if debug_mode:
        st.warning(f"Nenašiel som žiadne relevantné informácie o zákone {law_number}/{year} vo vektorovom úložisku")
    return None
