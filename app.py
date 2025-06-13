# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st
import openai
import logging
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import SystemMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)

# Set default encoding to UTF-8 without modifying stdout/stderr
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, '')  # Use system default locale
    except:
        pass  # Continue even if locale setting fails

# Set debug_mode to False by default
debug_mode = False

# Nastavenie slovenského jazyka a dark mode
st.set_page_config(
    page_title="Chatbot o bezpečnosti pri práci",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Aplikovanie dark mode pomocou CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        color: white !important;
    }
    .stApp {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

# Nadpis aplikácie
st.title("Chatbot o bezpečnosti pri práci")
st.write("Tento chatbot poskytuje informácie o slovenských zákonoch týkajúcich sa bezpečnosti pri práci.")
st.write("Upozornenie: Informácie poskytnuté týmto chatbotom nemusia byť vždy presné alebo aktuálne. Odporúčame ich overiť z dôveryhodných zdrojov.")

# Vytvorenie sidebar pre API kľúč
with st.sidebar:
    st.header("Nastavenia")
    # Store API key in session_state for persistence across reruns
    api_key = st.text_input("OpenAI API kľúč", type="password", value=st.session_state.get("api_key", ""))
    if api_key:
        st.session_state["api_key"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # Set env variable for OpenAI libraries
        st.success("API kľúč bol nastavený!")
    else:
        st.warning("Prosím, zadajte váš OpenAI API kľúč pre používanie chatbota.")
    
    # # Debug mode
    # debug_mode = st.checkbox("Debug režim", help="Zobraziť dodatočné informácie o vyhľadávaní")
    
    # Clear chat button
    if st.button("Vyčistiť chat", help="Vyčistiť históriu chatu a kontext konverzácie"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("## O aplikácii")
    st.markdown("Táto aplikácia využíva vektorovú databázu pre poskytovanie informácií o slovenských zákonoch o bezpečnosti pri práci.")

# Set OpenAI API key from session_state before any OpenAI/LLM/embedding code runs
if "api_key" in st.session_state and st.session_state["api_key"]:
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]

# Funkcia pre načítanie FAISS vektorového úložiska
@st.cache_resource
def load_vectorstore():
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            folder_path="processed_data",
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"Chyba pri načítavaní vektorového úložiska: {e}")
        return None

# Pomocná funkcia na hľadanie presných referencií zákona v obsahu dokumentov
def scan_document_for_law_reference(law_number, year, doc, debug_mode=False):
    """
    Skontroluje, či dokument obsahuje presnú referenciu na zákon
    """
    import re
    
    if not hasattr(doc, 'page_content'):
        return False
    
    content = doc.page_content.lower()
    # Skontrolujeme prvých 500 znakov, kde zvyčajne býva názov zákona
    first_part = content[:500]
    
    # Formáty na kontrolu
    formats_to_check = [
        f"{law_number}/{year}",
        f"{law_number}_{year}",
        f"{law_number}-{year}",
        f"{law_number}.{year}",
        f"{law_number}/{year} z.z",
        f"{law_number}_{year} z.z",
        f"{law_number}/{year} zb",
        # S medzerou medzi číslom a rokom
        f"{law_number} / {year}",
        f"{law_number} _ {year}",
        # Formát s č. pred číslom
        f"č. {law_number}/{year}",
        f"č. {law_number}_{year}",
        f"č.{law_number}/{year}"
    ]
    
    # Najprv kontrolujeme začiatok dokumentu, kde je najväčšia pravdepodobnosť nájsť meno zákona
    for format_str in formats_to_check:
        if format_str in first_part:
            if debug_mode:
                st.success(f"Našiel som zákon {format_str} v úvodnej časti dokumentu")
            return True
            
    # Potom kontrolujeme celý obsah - ale len s najpravdepodobnejšími formátmi
    main_formats = [f"{law_number}/{year}", f"{law_number}_{year}", f"{law_number}/{year} z.z", f"{law_number}_{year} z.z"]
    for format_str in main_formats:
        if format_str in content:
            if debug_mode:
                st.success(f"Našiel som zákon {format_str} v obsahu dokumentu")
            return True
            
    # Ešte kontrolujeme pomocou regulárnych výrazov
    regex_patterns = [
        rf"{law_number}\s*/\s*{year}",
        rf"{law_number}\s*_\s*{year}",
        rf"{law_number}\s*/\s*{year}\s+z\.z",
        rf"zákon\s+[^.]*\s+{law_number}\s*/\s*{year}",
        rf"vyhláška\s+[^.]*\s+{law_number}\s*/\s*{year}"
    ]
    
    for pattern in regex_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            if debug_mode:
                st.success(f"Našiel som zákon pomocou regex vzoru: {pattern}")
            return True
    
    return False

# Funkcia na spracovanie otázok o konkrétnych zákonoch
def handle_law_number_query(prompt, vectorstore, chat_history, debug_mode=False):
    # Detekcia otázok o konkrétnych zákonoch podľa čísla
    import re
    
    # Vzory na rozpoznanie odkazov na zákony ako "zákon 75/2023", "zákon č. 75/2023", "zákon 75_2023", atď.
    law_patterns = [
        r'(zákon|zákoník|predpis|nariadenie|vyhláška|zákonník)\s+(?:\w+\.)?\s*(\d+)[\/\.\-\_](\d+)',
        r'(?:zákon|zákoník|predpis|nariadenie|vyhláška|zákonník)?\s*(?:č\.|č|číslo)?\s*(\d+)[\/\.\-\_](\d+)',
        r'(\d+)[\/\.\-\_](\d+)(?:\s+Z\.z|\s+Zb|\s+Z\.z\.|\s+Zb\.)?',
        # Pridanie ďalších vzorov pre lepšiu detekciu
        r'(?:zákon|zákoník|predpis|nariadenie|vyhláška|zákonník)?\s*(\d+)\s+z\s+roku\s+(\d{4})',
        r'(?:zákon|zákoník|predpis|nariadenie|vyhláška|zákonník)\s+.*\s+(\d+)[\/\.\-\_](\d+)'
    ]
    
    # Kontrola, či otázka zodpovedá niektorému z vzorov pre číslo zákona
    law_number = None
    year = None
    for pattern in law_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        if matches:
            if len(matches[0]) == 3:  # Prvý vzor s slovom na začiatku
                law_number = matches[0][1]
                year = matches[0][2]
            elif len(matches[0]) == 2:  # Ostatné vzory
                law_number = matches[0][0]
                year = matches[0][1]
            break
    
    if not law_number or not year:
        # Nebolo detekované číslo zákona
        return None
    
    if debug_mode:
        st.write(f"Detekovaný zákon číslo: {law_number}/{year}")
        
    logging.info(f"Law number query detected: {law_number}/{year}")
    
    # Generovanie rôznych variácií formátu čísla zákona
    variations = [
        f"zákon {law_number}/{year}",
        f"zákon č. {law_number}/{year}",
        f"zákon č.{law_number}/{year}",
        f"{law_number}/{year}",
        f"zákon {law_number}-{year}",
        f"zákon {law_number} z roku {year}",
        f"zákon {law_number}.{year}",
        f"zákon číslo {law_number}/{year}",
        f"{law_number}_{year}",  # Formát s podčiarkovníkom pre súbory
        f"zákon {law_number}_{year}"  # Formát s podčiarkovníkom pre súbory s prefixom
    ]
    
    # Skúšanie každej variácie
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    for variation in variations:
        if debug_mode:
            st.write(f"Skúšam formát: {variation}")            # Priamy vyhľadávanie s touto variáciou
        try:
            specific_query = f"Čo presne upravuje {variation}? O čom je tento zákon?"
            direct_docs = vectorstore.similarity_search(specific_query, k=6)
            
            # Skontrolovať, či sa zákon nachádza v zdrojoch (metadátach) alebo obsahu
            found_in_metadata = False
            found_in_content = False
            
            for doc in direct_docs:
                # Kontrola metadát
                if hasattr(doc, 'metadata') and doc.metadata:
                    source_info = doc.metadata.get('source', '').lower()
                    
                    # Kontrola rôznych formátov zákona v metadátach
                    law_patterns_to_check = [
                        f"{law_number}/{year}", 
                        f"{law_number}_{year}",
                        f"{law_number}-{year}",
                        f"{law_number}.{year}"
                    ]
                    
                    for pattern in law_patterns_to_check:
                        if pattern in source_info:
                            found_in_metadata = True
                            if debug_mode:
                                st.write(f"Našiel som zákon v metadátach - formát: {pattern}")
                                st.write(f"Zdroj: {source_info}")
                            break
                
                # Dôkladná kontrola obsahu dokumentu
                found_in_content = scan_document_for_law_reference(law_number, year, doc, debug_mode)
                
                if found_in_metadata or found_in_content:
                    if debug_mode:
                        st.success(f"Našiel som zákon {law_number}/{year} v dokumentoch")
                    break
            
            if direct_docs or found_in_metadata:
                # Vytvorenie vlastného promptu zameraného na tento konkrétny zákon
                underscore_format = f"{law_number}_{year}"
                law_prompt = (
                    f"Hľadaj informácie o zákone {law_number}/{year} (môže sa nachádzať aj ako súbor s názvom {underscore_format}). "
                    f"Ak sa v dokumentoch nachádza tento zákon, informuj ma o čom presne je, "
                    f"čo upravuje a aké má hlavné body. Ak sa v dokumentoch tento zákon nenachádza, informuj ma o tom."
                )
                
                # Použitie QA reťazca na získanie úplnej odpovede
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
                    return_source_documents=True
                )
                
                # Pokus o získanie odpovede o tomto konkrétnom zákone
                response = qa_chain({"question": law_prompt, "chat_history": chat_history})
                
                # Ak neboli nájdené dobré výsledky, skúsime explicitné vyhľadávanie s podčiarkovníkom
                if not response["source_documents"] or any(phrase in response["answer"].lower() for phrase in [
                    "neviem", "nemám informácie", "nenašiel som", "nemám dostatok"
                ]):
                    # Skúsime explicitné vyhľadávanie s podčiarkovníkom
                    underscore_query = f"Informácie o {law_number}_{year}"
                    if debug_mode:
                        st.write(f"Skúšam priame vyhľadávanie s podčiarkovníkom: {underscore_query}")
                    
                    underscore_response = qa_chain({"question": underscore_query, "chat_history": chat_history})
                    if underscore_response["source_documents"] and not any(phrase in underscore_response["answer"].lower() for phrase in [
                        "neviem", "nemám informácie", "nenašiel som", "nemám dostatok"
                    ]):
                        return underscore_response
                
                # Kontrola, či odpoveď skutočne obsahuje informácie o zákone
                if response["source_documents"] and not any(phrase in response["answer"].lower() for phrase in [
                    "neviem", "nemám informácie", "nenašiel som", "nemám dostatok"
                ]):
                    return response
        except Exception as e:
            logging.error(f"Error in law search variation '{variation}': {str(e)}")
            if debug_mode:
                st.error(f"Chyba pri hľadaní: {str(e)}")
    
    # Ešte jeden posledný pokus - vyhľadávanie podľa kombinácie čísel a priame vyhľadávanie dokumentov
    try:
        # Najprv skúsime priame vyhľadávanie kombinácie čísel
        law_queries = [
            f"Obsahuje text číslo {law_number} a rok {year}",
            f"zákon obsahujúci číslo {law_number} a {year}",
            f"právny predpis {law_number} {year}"
        ]
        
        for query in law_queries:
            if debug_mode:
                st.write(f"Posledný pokus vyhľadávania: {query}")
            
            final_response = qa_chain({"question": query, "chat_history": chat_history})
            if final_response["source_documents"] and not any(phrase in final_response["answer"].lower() for phrase in [
                "neviem", "nemám informácie", "nenašiel som", "nemám dostatok"
            ]):
                # Vytvorme lepšiu odpoveď, ktorá naozaj odpovedá na pôvodnú otázku
                better_prompt = f"Na základe týchto dokumentov, poskytni informácie o zákone {law_number}/{year} " + \
                               f"(alebo {law_number}_{year}). O čom presne je tento zákon a čo upravuje?"
                
                better_response = qa_chain({"question": better_prompt, "chat_history": []})
                return better_response

        # Potom skúsime priame vyhľadávanie v dokumentoch - optimalizované
        direct_docs = vectorstore.similarity_search(f"{law_number}_{year}", k=5)  # Znížené z 8
        for doc in direct_docs:
            content = doc.page_content.lower() if hasattr(doc, 'page_content') else ''
            source = doc.metadata.get('source', '').lower() if hasattr(doc, 'metadata') and doc.metadata else ''
            
            # Hľadáme akékoľvek zmienky o zákone
            if (f"{law_number}_{year}" in content or f"{law_number}_{year}" in source or
                f"{law_number}/{year}" in content or f"{law_number}/{year}" in source):
                if debug_mode:
                    st.write(f"Našiel som priamy odkaz na zákon v dokumente: {source}")
                
                # Použijeme tento dokument na vytvorenie odpovede
                direct_prompt = f"Na základe tohto dokumentu, poskytni súhrn zákona {law_number}/{year}: {content[:1000]} " + \
                               f"O čom presne je tento zákon a čo upravuje?"
                
                direct_response = qa_chain({"question": direct_prompt, "chat_history": []})
                return direct_response
    except Exception as e:
        logging.error(f"Error in final law search attempt: {str(e)}")
        if debug_mode:
            st.error(f"Chyba pri poslednom pokuse: {str(e)}")

    # Posledný pokus - hľadanie pomocou "obsahovej zhody" dokumentov
    try:
        # Použijeme raw vyhľadávanie obsahov dokumentov
        content_search_query = f"zákony právne predpisy {law_number} {year}"
        if debug_mode:
            st.write(f"Posledný pokus - obsahové vyhľadávanie: {content_search_query}")
        
        raw_docs = vectorstore.similarity_search(content_search_query, k=8)  # Znížené z 10
        
        # Manuálne filtrujeme nájdené dokumenty
        matching_docs = []
        for doc in raw_docs:
            content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
            
            # Kontrola obsahu na presné znenie čísla zákona
            if (f"{law_number}/{year}" in content or 
                f"{law_number}_{year}" in content or 
                f"{law_number}-{year}" in content or 
                f"{law_number}.{year}" in content or 
                f"{law_number}/{year} z.z" in content or 
                f"{law_number}_{year} z.z" in content):
                
                matching_docs.append(doc)
                if debug_mode:
                    st.success(f"Našiel som zákon v obsahovom vyhľadávaní")
        
        if matching_docs:
            # Vytvoríme nový reťazec pre odpoveď len s vyfiltrovanými dokumentami
            qa_with_matching = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.2, model_name="gpt-4o"),
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            content_prompt = f"Poskytni informácie o zákone {law_number}/{year} (alebo {law_number}_{year}) na základe týchto dokumentov. O čom je tento zákon a čo upravuje?"
            
            content_response = qa_with_matching({"question": content_prompt, "chat_history": []})
            return content_response
    except Exception as e:
        logging.error(f"Error in content search for law: {str(e)}")
        if debug_mode:
            st.error(f"Chyba pri obsahovom vyhľadávaní: {str(e)}")
    
    # Ak sme sa dostali sem, nenašli sme dobré zodpovedanie
    if debug_mode:
        st.warning(f"Nenašiel som informácie o zákone {law_number}/{year} ani {law_number}_{year}")
    
    return None

# Funkcia pre priame vyhľadávanie zákonov v textovom obsahu
def direct_law_search(law_number, year, vectorstore, debug_mode=False):
    """
    Funkcia na priame vyhľadávanie zákonov v texte dokumentov
    """
    import re
    
    # Vyskúšame rôzne formáty zákona
    formats = [
        f"{law_number}_{year}",
        f"{law_number}/{year}",
        f"{law_number}-{year}",
        f"{law_number}.{year}",
        f"zákon {law_number}_{year}",
        f"zákon {law_number}/{year}",
        f"zákon č. {law_number}/{year}",
        # Pridáme formát so Z.z. a Zb. príponou, ktorá sa často nachádza v zákonoch
        f"{law_number}/{year} Z.z.",
        f"{law_number}_{year} Z.z.",
        f"{law_number}/{year} Zb."
    ]
    
    # Skúsime najprv optimalizovaný formát, ktorý kombinuje oba hlavné formáty a umožňuje rôzne variácie
    combined_format = f"{law_number}/{year} OR {law_number}_{year} OR zákon {law_number}"
    if debug_mode:
        st.write(f"Skúšam kombinovaný formát: {combined_format}")
    
    try:
        # Najprv skúsime priamy MMR retrieval pre lepšie výsledky na vektorovom úložisku
        try:
            mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 15})
            docs = mmr_retriever.get_relevant_documents(combined_format)
            if debug_mode and docs:
                st.success(f"Našiel som dokumenty pomocou MMR retrieval v vektorovom úložisku")
        except Exception as mmr_error:
            if debug_mode:
                st.error(f"Chyba pri MMR vyhľadávaní: {str(mmr_error)}")
            # Ak MMR zlyhá, použijeme štandardné vyhľadávanie
            docs = vectorstore.similarity_search(combined_format, k=8)
        
        # Kontrolujeme, či niektorý z dokumentov obsahuje zákon v prvých riadkoch
        if docs:
            for doc in docs:
                content = doc.page_content if hasattr(doc, 'page_content') else ""
                # Pozrieme sa na prvých 200 znakov, kde zvyčajne je názov zákona
                first_part = content[:200].lower()
                
                # Kontrola označenia zákona v hlavičke dokumentu
                for format_check in [f"{law_number}/{year}", f"{law_number}_{year}", f"{law_number}-{year}"]:
                    if format_check in first_part:
                        if debug_mode:
                            st.success(f"Našiel som zákon v hlavičke dokumentu: {format_check}")
                        return docs
    except Exception as e:
        if debug_mode:
            st.error(f"Chyba pri kombinovanom vyhľadávaní: {str(e)}")
    
    # Pokračujeme štandardným vyhľadávaním
    for format_str in formats:
        if debug_mode:
            st.write(f"Skúšam formát: {format_str}")
            
        # Priame vyhľadávanie textu
        try:
            docs = vectorstore.similarity_search(format_str, k=5)
            
            if docs:
                # Kontrola či dokument naozaj obsahuje hľadaný zákon
                for doc in docs:
                    content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
                    
                    # Hľadáme zákon v obsahu - rozšírené hľadanie s pripojenými Z.z./Zb. formátmi
                    law_formats_to_check = [
                        f"{law_number}_{year}", 
                        f"{law_number}/{year}", 
                        f"{law_number}-{year}", 
                        f"zákon.*{law_number}",
                        f"{law_number}/{year}\s*z\.z",
                        f"{law_number}_{year}\s*z\.z",
                        f"{law_number}/\s*{year}\s*zb"
                    ]
                    
                    for law_format in law_formats_to_check:
                        if re.search(law_format, content, re.IGNORECASE):
                            if debug_mode:
                                st.success(f"Našiel som zákon v obsahu dokumentu použitím formátu: {law_format}")
                                # Zobrazíme prvú časť obsahu, kde sa našiel zákon
                                match = re.search(law_format, content, re.IGNORECASE)
                                if match:
                                    start_index = max(0, match.start() - 20)
                                    end_index = min(len(content), match.end() + 20)
                                    st.write(f"Kontext nálezu: ...{content[start_index:end_index]}...")
                            return docs
                            
                    # Tiež hľadáme v metadátach
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source = doc.metadata.get('source', '').lower()
                        if any(law_format in source for law_format in [f"{law_number}_{year}", f"{law_number}/{year}"]):
                            if debug_mode:
                                st.success(f"Našiel som zákon v metadátach: {source}")
                            return docs
        except Exception as e:
            if debug_mode:
                st.error(f"Chyba pri priamom vyhľadávaní: {str(e)}")                                # Skúsme ešte vyhľadávanie na základe čistej zhody reťazca v dokumentoch
    try:
        # Získame dokumenty bez predselekcií, ale použijeme kombináciu užívateľského promptu a informácií o zákone
        combined_query = f"{prompt} {law_number}/{year} {law_number}_{year} zákony legislatíva"
        if debug_mode:
            st.info(f"Skúšam kombinované vyhľadávanie: {combined_query}")
        
        # Použijeme MMR vyhľadávanie pre lepšiu rozmanitosť výsledkov - optimalizované
        try:
            mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 15, "lambda_mult": 0.7})
            raw_docs = mmr_retriever.get_relevant_documents(combined_query)
            if debug_mode and raw_docs:
                st.success(f"Našiel som {len(raw_docs)} dokumentov pomocou MMR vyhľadávania")
        except Exception as mmr_err:
            if debug_mode:
                st.error(f"MMR vyhľadávanie zlyhalo: {str(mmr_err)}, skúšam štandardné vyhľadávanie")
            # Fallback na štandardné vyhľadávanie
            raw_docs = vectorstore.similarity_search(combined_query, k=10, score_threshold=0.28)  # Optimalizované hodnoty
        
        for doc in raw_docs:
            content = doc.page_content.lower() if hasattr(doc, 'page_content') else ""
            
            # Hľadáme presné ID zákona v obsahu
            for exact_id in [f"{law_number}/{year}", f"{law_number}_{year}"]:
                if exact_id in content:
                    if debug_mode:
                        st.success(f"Našiel som zákon presnou zhodou reťazca: {exact_id}")
                    return [doc]  # Vrátime tento dokument ako výsledok
    except Exception as e:
        if debug_mode:
            st.error(f"Chyba pri hľadaní presnej zhody: {str(e)}")
    
    # Ak sa dostaneme sem, skúsime priame vyhľadávanie súborov v adresári stiahnute_zakony
    try:
        # Import funkciu pre priame vyhľadávanie súborov
        from enhanced_law_search import direct_file_search
        
        if debug_mode:
            st.write(f"Skúšam nájsť informácie o zákone {law_number}/{year} vo vektorovom úložisku")
            
        # Priame vyhľadávanie bude pracovať iba s vektorovým úložiskom, pretože pôvodný adresár neexistuje
        direct_result = direct_file_search(law_number, year, debug_mode)
        
        # Táto podmienka už nikdy nebude splnená, pretože direct_file_search vždy vráti None,
        # ale ponechávame ju pre zachovanie štruktúry kódu
        if direct_result:
            # Vytvorenie Document objektu
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.schema.document import Document
            
            doc = Document(
                page_content=direct_result["content"],
                metadata={"source": f"vectorstore/{os.path.basename(direct_result['file_path'])}"}
            )
            if debug_mode:
                st.success(f"Našiel som zákon {law_number}/{year}")
            return [doc]
    except Exception as e:
        if debug_mode:
            st.error(f"Chyba pri vyhľadávaní: {str(e)}")
    
    # Ak sa dostaneme sem, nenašli sme žiadne dokumenty
    if debug_mode:
        st.warning(f"Nenašiel som žiadne dokumenty obsahujúce zákon {law_number}/{year} alebo {law_number}_{year}")
    return None

# Inicializácia chat histórie
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zobrazenie predchádzajúcich správ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Spracovanie nových správ
if "api_key" in st.session_state and st.session_state["api_key"]:
    api_key = st.session_state["api_key"]
    # Načítanie vektorového úložiska
    vectorstore = load_vectorstore()
    
    if vectorstore:
        # Vytvorenie chatbota s dobrou rovnováhou medzi kreativitou a presnosťou
        llm = ChatOpenAI(
            temperature=0.4,  # Nižšia teplota pre lepšiu presnosť
            model_name="gpt-4o"
        )
        
        # Vytvoriť reťazec s vlastným systémovým promptom pre faktické odpovede
        try:
            # Skúšame najprv s MMR retrieverom pre lepšiu diverzitu výsledkov
            mmr_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,  # Počet dokumentov na vrátenie - znížené z 8 pre lepšiu efektivitu
                    "fetch_k": 12,  # Počet dokumentov na fetch pred skoringom - znížené z 15
                    "lambda_mult": 0.7,  # Vyváženie medzi relevantnosťou a diverzitou (0.0-1.0)
                    "score_threshold": 0.28,  # Zvýšený threshold pre lepšiu relevantnosť
                }
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=mmr_retriever,
                return_source_documents=True,
                verbose=False
            )
            if debug_mode:
                st.success("Použitý MMR retriever pre lepšie výsledky")
        except Exception as retriever_error:
            # Fallback na štandardný retriever ak MMR nefunguje
            if debug_mode:
                st.warning(f"MMR retriever zlyhalo: {str(retriever_error)}. Použitý štandardný retriever.")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={
                    "k": 6,  # Znížené z 8 pre lepšiu efektivitu
                    "score_threshold": 0.28,  # Zvýšený threshold pre relevantnejšie výsledky
                }),
                return_source_documents=True,
                verbose=False
            )
        
        # Chat interface
        if prompt := st.chat_input("Napíšte vašu otázku o bezpečnosti pri práci..."):
            # Pridanie užívateľskej správy do histórie
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Zobrazenie užívateľskej správy
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Zobrazenie indikátora načítavania
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Premýšľam...")
                
                # Získanie chat histórie pre kontext
                chat_history = [(q["content"], a["content"]) for q, a in zip(
                    [m for m in st.session_state.messages if m["role"] == "user"][:-1],
                    [m for m in st.session_state.messages if m["role"] == "assistant"]
                )]
                
                try:
                    # If in debug mode, first show similar document contents and embedding info
                    if debug_mode:
                        with st.expander("Debug: Podobné dokumenty"):
                            st.markdown("### Embedding informácie")
                            st.info(f"Vstupný prompt na embedding: '{prompt}'")
                            
                            # Test embedding creation
                            try:
                                embeddings = OpenAIEmbeddings()
                                # Just check that we can create embeddings
                                test_embedding = embeddings.embed_query(prompt)
                                st.success(f"Embedding vytvorený - dĺžka vektora: {len(test_embedding)}")
                            except Exception as embed_error:
                                st.error(f"Chyba pri vytváraní embeddings: {str(embed_error)}")
                            
                            st.markdown("### Výsledky vyhľadávania")
                            # Try both similarity search and mmr search
                            st.markdown("#### Similarity Search:")
                            try:
                                raw_docs = vectorstore.similarity_search(prompt, k=4)  # Znížené pre lepšiu efektivitu
                                if not raw_docs:
                                    st.warning("Similarity search nenašiel žiadne dokumenty")
                                else:
                                    st.success(f"Nájdených {len(raw_docs)} dokumentov")
                                    for i, doc in enumerate(raw_docs):
                                        st.markdown(f"**Dokument {i+1}:**")
                                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                        st.markdown(f"Zdroj: {doc.metadata.get('source', 'Neznámy')}")
                                        st.markdown("---")
                            except Exception as e:
                                st.error(f"Chyba pri similarity search: {str(e)}")
                                
                            st.markdown("#### MMR Search (pre diverzitu výsledkov):")
                            try:
                                mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
                                mmr_docs = mmr_retriever.get_relevant_documents(prompt)
                                if not mmr_docs:
                                    st.warning("MMR search nenašiel žiadne dokumenty")
                                else:
                                    st.success(f"MMR nájdených {len(mmr_docs)} dokumentov")
                                    for i, doc in enumerate(mmr_docs):
                                        st.markdown(f"**Dokument {i+1}:**")
                                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                        st.markdown(f"Zdroj: {doc.metadata.get('source', 'Neznámy')}")
                                        st.markdown("---")
                            except Exception as mmr_e:
                                st.error(f"Chyba pri MMR search: {str(mmr_e)}")
                    
                    # Process original query
                    strict_prompt = (
                        prompt +
                        "Answer concisely and in Slovak.. " 
                        "Use the information that is most semantically related to the question, even if it is not exactly the same.. "
                        "Search for information in all available sources about laws and work safety. "
                        "If you don't have enough information, state it clearly."
                        "Always mention which law or document you are referring to, if applicable."
                    )
                    response = qa_chain({"question": strict_prompt, "chat_history": chat_history})
                    answer = response["answer"]
                    
                    # If no results found, try with reformulated query
                    fallback_phrases = [
                        "neviem", "nemám dostatok informácií", "neviem odpovedať", "nemám informácie", 
                        "I don't know", "No relevant information found"
                    ]
                    
                    if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                        # Try MMR retrieval instead of similarity search
                        if debug_mode:
                            st.markdown("**Debug: Skúšam MMR retrieval**")
                        
                        try:
                            mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
                            mmr_docs = mmr_retriever.get_relevant_documents(strict_prompt)
                            
                            if mmr_docs:
                                mmr_response = qa_chain({"question": strict_prompt, "chat_history": chat_history})
                                if mmr_response["source_documents"] and not any(phrase in mmr_response["answer"].lower() for phrase in fallback_phrases):
                                    response = mmr_response
                                    answer = mmr_response["answer"]
                                    
                                    if debug_mode:
                                        st.success("MMR retrieval poskytol relevantné dokumenty")
                        except Exception as retrieval_error:
                            if debug_mode:
                                st.error(f"Chyba pri MMR retrieval: {str(retrieval_error)}")
                        
                        # If still no results, try with reformulated query
                        if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                            # Create a reformulation prompt
                            reformulation_prompt = (
                                f"Otázka: {prompt}\n\n"
                                "Preformuluj túto otázku na 2-3 alternatívne verzie, aby lepšie vyhovovala "
                                "vyhľadávaniu v databáze o zákonoch a bezpečnosti práce. "
                                "Vráť len alternatívne otázky oddelené znakom '|', bez ďalších komentárov."
                            )
                            
                            # Get reformulated queries
                            reformulation_llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
                            reformulation_response = reformulation_llm.invoke(reformulation_prompt)
                            alternative_queries = reformulation_response.content.split('|')
                            
                            # Try each alternative query
                            for alt_query in alternative_queries:
                                alt_query = alt_query.strip()
                                if alt_query:
                                    alt_response = qa_chain({
                                        "question": alt_query + "\n\nOdpovedaj stručne a v slovenčine.",
                                        "chat_history": chat_history
                                    })
                                    
                                    if alt_response["source_documents"] and not any(phrase in alt_response["answer"].lower() for phrase in fallback_phrases):
                                        response = alt_response
                                        answer = alt_response["answer"]
                                        break
                            
                            # If still no good results, use fallback message
                            if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                                answer = "Nenašiel som konkrétne informácie. Skúste sa opýtať na konkrétny zákon alebo tému súvisiacu s bezpečnosťou práce, napríklad: 'Čo hovorí Zákonník práce o osobných ochranných prostriedkoch?'"
                    
                    # If reformulation doesn't work, try keyword extraction
                    if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                        if debug_mode:
                            st.markdown("**Debug: Skúšam vyhľadávanie kľúčových slov**")
                        
                        # Extract keywords
                        keyword_prompt = (
                            f"Otázka: {prompt}\n\n"
                            "Extrahuj 3-5 kľúčových slov z tejto otázky, ktoré by mohli byť užitočné pre vyhľadávanie "
                            "v databáze zákonov o bezpečnosti práce. Vráť len samotné kľúčové slová oddelené čiarkou, bez úvodzoviek či iných znakov."
                        )
                        
                        keyword_llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")
                        keyword_response = keyword_llm.invoke(keyword_prompt)
                        keywords = [kw.strip() for kw in keyword_response.content.split(',')]
                        
                        if debug_mode:
                            st.markdown(f"Extrahované kľúčové slová: {', '.join(keywords)}")
                        
                        # Try each keyword
                        for keyword in keywords:
                            if len(keyword) < 3:  # Skip very short keywords
                                continue
                                
                            keyword_query = f"Informácie o téme: {keyword}"
                            keyword_response = qa_chain({
                                "question": keyword_query + "\n\nOdpovedaj stručne a v slovenčine.",
                                "chat_history": []
                            })
                            
                            if keyword_response["source_documents"] and not any(phrase in keyword_response["answer"].lower() for phrase in fallback_phrases):
                                # Create a focused answer using this keyword and original question
                                focused_query = (
                                    f"Otázka: {prompt}\n\n"
                                    f"Použitím informácií o '{keyword}', odpovedz na túto otázku "
                                    f"stručne a v slovenčine."
                                )
                                
                                focused_response = qa_chain({
                                    "question": focused_query, 
                                    "chat_history": []
                                })
                                
                                if focused_response["source_documents"]:
                                    response = focused_response
                                    answer = focused_response["answer"]
                                    
                                    if debug_mode:
                                        st.success(f"Vyhľadávanie kľúčového slova '{keyword}' poskytlo relevantné dokumenty")
                                    break
                        
                        # If still no good results, use fallback message
                        if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                            # Try to provide a better fallback with suggestions
                            common_topics = [
                                "osobné ochranné prostriedky",
                                "pracovné úrazy",
                                "bezpečnostné školenie",
                                "zákonník práce",
                                "pracovné podmienky",
                                "pracovná zdravotná služba"
                            ]
                            
                            fallback_suggestions = "\n\nSkúste sa opýtať na niektorú z týchto tém:\n"
                            fallback_suggestions += "\n".join([f"- {topic}" for topic in common_topics])
                            
                            answer = f"Žiaľ, nenašiel som konkrétne informácie k vašej otázke. {fallback_suggestions}"
                            
                            # Try to extract the main topic for additional help
                            try:
                                topic_prompt = f"Identifikuj hlavnú tému otázky v jednom slove: '{prompt}'"
                                topic_llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")
                                topic_response = topic_llm.invoke(topic_prompt)
                                main_topic = topic_response.content.strip()
                                
                                if main_topic and len(main_topic) > 2:
                                    answer += f"\n\nPre tému '{main_topic}' skúste špecifickejšiu formuláciu otázky."
                            except:
                                pass
                    
                    # # Pridanie informácií o zdrojoch (unikátne)
                    # if response["source_documents"]:
                    #     sources = set()
                    #     for doc in response["source_documents"]:
                    #         if hasattr(doc, 'metadata') and doc.metadata:
                    #             source_info = doc.metadata.get('source', 'Neznámy zdroj')
                    #             sources.add(source_info)
                    #     if sources:
                    #         answer += "\n\n**Zdroje:**"
                    #         for source in sources:
                    #             answer += f"\n- {source}"
                    
                    # Špeciálne spracovanie otázok o zákonoch podľa čísla
                    law_response = handle_law_number_query(prompt, vectorstore, chat_history, debug_mode)
                    
                    # Ak bežné spracovanie zlyhalo ale otázka obsahuje čísla, ktoré môžu byť zákonom
                    if not law_response:
                        import re
                        
                        # Importujeme modul s vylepšeným vyhľadávaním zákonov
                        try:
                            # Import enhanced law search module (will only execute once)
                            from enhanced_law_search import direct_law_search, get_law_content
                            if debug_mode:
                                st.write("Načítaný vylepšený vyhľadávací modul pre zákony")
                        except ImportError as ie:
                            if debug_mode:
                                st.error(f"Nepodarilo sa načítať modul enhanced_law_search: {str(ie)}")
                        
                        # Najprv skúsime priamo vyhľadať konkrétne číslo zákona v obsahu dokumentov
                        # Hľadáme explicitne zákon 75/2023 alebo 75_2023 atď. v PROCESOVANÝCH DÁTACH (nie v súboroch)
                        direct_matches = re.findall(r'(\d+)\s*[\/\._-]?\s*(\d{4}|\d{2})', prompt)
                        
                        if direct_matches:
                            for num, year in direct_matches:
                                # Kontrola či rok vyzerá platne
                                if len(year) == 2:  # Ak je rok dvojciferný, pripojíme 20 alebo 19
                                    if int(year) > 50:  # Staršie zákony, 19xx
                                        year = f"19{year}"
                                    else:  # Novšie zákony, 20xx
                                        year = f"20{year}"
                                
                                if debug_mode:
                                    st.write(f"Skúšam priame vyhľadávanie pre zákon: {num}/{year}")
                                
                                if debug_mode:
                                    st.write(f"Vyhľadávanie zákona {num}/{year} vo vektorovom úložisku (processed_data)")
                                
                                # Skúsime použiť vylepšenú funkciu na priame vyhľadávanie v vektorovom úložisku
                                try:
                                    # Použijeme novú funkciu ak je dostupná - je efektívnejšia
                                    if 'get_law_content' in locals() or 'get_law_content' in globals():
                                        # Použijeme kompletné vyhľadávanie s vylepšeným modulom
                                        law_response = get_law_content(num, year, vectorstore, chat_history, debug_mode)
                                        if law_response:
                                            answer = law_response["answer"]
                                            if debug_mode:
                                                st.success(f"Našiel som zákon {num}/{year} pomocou vylepšeného vyhľadávacieho modulu")
                                            continue
                                    
                                    # Fallback na pôvodnú funkciu
                                    direct_docs = direct_law_search(num, year, vectorstore, debug_mode)
                                    
                                    if direct_docs:
                                        # Vytvoríme nový kontext pre odpoveď
                                        direct_prompt = (
                                            f"Na základe poskytnutých dokumentov, vysvetli čo presne upravuje zákon {num}/{year} alebo {num}_{year}? "
                                            f"O čom je tento zákon, aké sú jeho hlavné body? "
                                            f"Ak sa v dokumentoch nenachádza tento zákon, povedz mi to."
                                        )
                                        
                                        # Vytvoríme nový chain s parametrami na presné vyhľadávanie
                                        focused_chain = ConversationalRetrievalChain.from_llm(
                                            llm=ChatOpenAI(temperature=0.2, model_name="gpt-4o"),
                                            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Znížené z 8
                                            return_source_documents=True
                                        )
                                except Exception as search_error:
                                    if debug_mode:
                                        st.error(f"Chyba pri vyhľadávaní zákona: {str(search_error)}")
                                    # Continue with standard search if enhanced search fails
                                    direct_docs = vectorstore.similarity_search(f"{num}_{year} OR {num}/{year}", k=8)
                                    
                                    # Pokúsime sa získať presnú odpoveď a ukončiť hľadanie
                                    direct_response = focused_chain({"question": direct_prompt, "chat_history": []})
                                    
                                    if direct_response["source_documents"] and not any(phrase in direct_response["answer"].lower() for phrase in 
                                        ["neviem", "nemám informácie", "nenašiel som", "nemám dostatok"]):
                                        
                                        law_response = direct_response
                                        answer = direct_response["answer"]
                                        if debug_mode:
                                            st.success(f"Našiel som zákon pomocou priameho vyhľadávania v obsahu: {num}/{year}")
                                        break
                        
                        # Skúsime vyhľadať v obsahu s rôznymi formátmi
                        if not law_response:
                            # Hľadanie dvoch čísel blízko seba (číslo zákona a rok)
                            potential_laws = re.findall(r'(\d+)\s*[\/_\.-]?\s*(\d{4}|\d{2})', prompt)
                            
                            for num, year in potential_laws:
                                # Kontrola či rok vyzerá platne
                                if len(year) == 2:  # Ak je rok dvojciferný, pripojíme 20 alebo 19
                                    if int(year) > 50:  # Staršie zákony, 19xx
                                        year = f"19{year}"
                                    else:  # Novšie zákony, 20xx
                                        year = f"20{year}"
                                        
                                if debug_mode:
                                    st.write(f"Detekovaný potenciálny zákon: {num}/{year}")
                                
                                # Priamo použijeme funkciu direct_law_search pre komplexné vyhľadávanie
                                direct_docs = direct_law_search(num, year, vectorstore, debug_mode)
                                
                                # Ak sme našli dokumenty, spracujeme ich
                                if direct_docs:
                                    # Vytvoríme nový kontext pre odpoveď
                                    direct_prompt = (
                                        f"Na základe poskytnutých dokumentov, vysvetli čo presne upravuje zákon {num}/{year} alebo {num}_{year}? "
                                        f"O čom je tento zákon, aké sú jeho hlavné body? "
                                        f"Ak sa v dokumentoch nenachádza tento zákon, povedz mi to."
                                    )
                                    
                                    # Použijeme ConversationalRetrievalChain pre lepšie výsledky vyhľadávania
                                    focused_chain = ConversationalRetrievalChain.from_llm(
                                        llm=ChatOpenAI(temperature=0.2, model_name="gpt-4o"),
                                        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Znížené z 8
                                        return_source_documents=True
                                    )
                                    
                                    # Pokúsime sa získať presnú odpoveď
                                    direct_response = focused_chain({"question": direct_prompt, "chat_history": []})
                                    
                                    # Overíme, či odpoveď obsahuje relevantné informácie
                                    if direct_response["source_documents"] and not any(phrase in direct_response["answer"].lower() for phrase in 
                                        ["neviem", "nemám informácie", "nenašiel som", "nemám dostatok"]):
                                        
                                        law_response = direct_response
                                        answer = direct_response["answer"]
                                        if debug_mode:
                                            st.success(f"Našiel som zákon pomocou priameho vyhľadávania v obsahu: {num}/{year} alebo {num}_{year}")
                                        break
                                
                                # Ak sme nenašli dokumenty, vyskúšame ešte priame vyhľadávanie rôznych formátov
                                if not law_response:
                                    # Vyskúšame rôzne formáty pre priame vyhľadávanie
                                    for format_char in ['_', '/', '-', '.']:
                                        law_id = f"{num}{format_char}{year}"
                                        
                                        if debug_mode:
                                            st.write(f"Skúšam priamo hľadať: {law_id}")
                                        
                                        try:
                                            # Hľadáme dokumenty obsahujúce presný formát - znížený počet pre efektivitu
                                            direct_docs = vectorstore.similarity_search(law_id, k=4)
                                            
                                            # Ak sme niečo našli, spracujeme to
                                            if direct_docs:
                                                # Vytvoríme nový kontext pre odpoveď
                                                direct_prompt = f"Na základe týchto dokumentov, čo upravuje zákon {num}/{year} alebo {num}_{year}?"
                                                direct_response = qa_chain({"question": direct_prompt, "chat_history": []})
                                                
                                                # Validujeme odpoveď - len ak je relevantná
                                                if direct_response["source_documents"] and not any(phrase in direct_response["answer"].lower() for phrase in 
                                                    ["neviem", "nemám informácie", "nenašiel som", "nemám dostatok"]):
                                                    
                                                    law_response = direct_response
                                                    answer = direct_response["answer"]
                                                    if debug_mode:
                                                        st.success(f"Našiel som zákon pomocou priameho vyhľadávania: {law_id}")
                                                    break  # Prerušíme vyhľadávanie keď nájdeme dobrú odpoveď
                                        except Exception as e:
                                            if debug_mode:
                                                st.error(f"Chyba pri priamom vyhľadávaní: {str(e)}")
                                    
                                # Ak sme našli odpoveď, ukončíme cyklus
                                if law_response:
                                    break
                    
                    # Po skončení všetkých vyhľadávaní, nastavíme výsledky
                    if law_response:
                        # Ak sme našli zákon použijeme odpoveď z law_response
                        response = law_response
                        answer = law_response["answer"]
                        
                        # # Pridanie informácií o zdrojoch (unikátne)
                        # if law_response["source_documents"]:
                        #     sources = set()
                        #     for doc in law_response["source_documents"]:
                        #         if hasattr(doc, 'metadata') and doc.metadata:
                        #             source_info = doc.metadata.get('source', 'Neznámy zdroj')
                        #             sources.add(source_info)
                        #     if sources:
                        #         answer += "\n\n**Zdroje:**"
                        #         for source in sources:
                        #             answer += f"\n- {source}"
                        
                        if debug_mode:
                            st.success("Úspešne sa našiel a spracoval zákon")
                    else:
                        # Extrahujeme číslo zákona z otázky pre lepšiu odozvu
                        import re
                        law_match = re.search(r'(\d+)[\/\.\-_](\d{4}|\d{2})', prompt)
                        if law_match:
                            law_number = law_match.group(1)
                            year = law_match.group(2)
                            
                            # Kontrola či rok vyzerá platne ako 4-ciferný
                            if len(year) == 2:
                                pass  # TODO: Add logic if needed
                            
                            # Ak bola otázka explicitne o zákone, ale nenašli sme ho, pridáme ďalšie informácie
                            if "zákon" in prompt.lower():
                                # Pokus o priame vyhľadávanie v metadátach dokumentov
                                try:
                                    # Vyhľadáme súbory, ktoré by mohli obsahovať tento zákon - optimalizované
                                    direct_docs = vectorstore.similarity_search(f"{law_number}_{year} OR {law_number}/{year}", k=3)
                                    sources_found = []
                                    
                                    for doc in direct_docs:
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            source = doc.metadata.get('source', '')
                                            if f"{law_number}_{year}" in source or f"{law_number}/{year}" in source:
                                                sources_found.append(source)
                                    
                                    # Ak sme našli zákon v metadátach
                                    if sources_found:
                                        answer += f"\n\nPoznámka: Našiel som zákon {law_number}/{year} v nasledujúcich súboroch, " + \
                                                f"ale nepodarilo sa mi extrahovať informácie: {', '.join(sources_found)}. " + \
                                                f"Skúste použiť Debug režim pre viac informácií."
                                    else:
                                        answer += f"\n\nPoznámka: Vyhľadávanie zákona {law_number}/{year} (alebo {law_number}_{year}) bolo neúspešné. " + \
                                                f"Skúste iný formát čísla zákona (napr. 'zákon {law_number}_{year}') alebo použite Debug režim pre viac informácií."
                                except Exception as e:
                                    if debug_mode:
                                        st.error(f"Chyba pri overovaní metadát: {str(e)}")
                                    answer += f"\n\nPoznámka: Vyhľadávanie zákona {law_number}/{year} (alebo {law_number}_{year}) bolo neúspešné. " + \
                                            f"Skúste iný formát čísla zákona alebo použite Debug režim pre viac informácií."
                except Exception as e:
                    # Log the error and show a friendly message
                    logging.error(f"Error processing query: {str(e)}")
                    answer = "Nastala chyba pri spracovaní vašej otázky. Prosím, skúste to znova alebo formulujte vašu otázku inak."
                    if debug_mode:
                        st.error(f"Chyba pri spracovaní: {str(e)}")
                
                # Zobrazenie odpovede
                message_placeholder.markdown(answer)
                
                # Pridanie správy asistenta do histórie
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # # Uloženie stavu po spracovaní
                # # (nie je nutné, ale môže pomôcť pri obnove stavu v prípade chyby)
                # with st.expander("Debug: Interné správy", expanded=True):
                #     st.write("### Pôvodná otázka")
                #     st.markdown(prompt)
                #     st.write("### Spracovaná otázka")
                #     st.markdown(strict_prompt)
                #     st.write("### Získané odpovede")
                #     for i, doc in enumerate(response.get("source_documents", [])):
                #         st.write(f"**Dokument {i+1}:** {doc.metadata.get('source', 'Neznámy')}")
                #         st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                #         st.markdown("---")
                #     if debug_mode:
                #         st.write("### Debug informácie")
                #         st.write(f"Detekované číslo zákona: {law_number}/{year}")
                #         st.write(f"Počet nájdených dokumentov: {len(response.get('source_documents', []))}")
                #         if law_response:
                #             st.write("Zákon bol úspešne nájdený a spracovaný.")
                #         else:
                #             st.write("Zákon nebol nájdený, aj keď bola otázka o zákone.")
