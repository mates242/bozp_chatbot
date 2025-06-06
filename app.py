# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st
import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Set default encoding to UTF-8 without modifying stdout/stderr
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, '')  # Use system default locale
    except:
        pass  # Continue even if locale setting fails

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

# Vytvorenie sidebar pre API kľúč
with st.sidebar:
    st.header("Nastavenia")
    api_key = st.text_input("OpenAI API kľúč", type="password")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API kľúč bol nastavený!")
    else:
        st.warning("Prosím, zadajte váš OpenAI API kľúč pre používanie chatbota.")
    
    st.markdown("---")
    st.markdown("## O aplikácii")
    st.markdown("Táto aplikácia využíva vektorovú databázu pre poskytovanie informácií o slovenských zákonoch o bezpečnosti pri práci.")

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

# Inicializácia chat histórie
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zobrazenie predchádzajúcich správ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Spracovanie nových správ
if api_key:
    # Načítanie vektorového úložiska
    vectorstore = load_vectorstore()
    
    if vectorstore:
        # Vytvorenie chatbota s vyššou kreativitou ale stále faktickou presnosťou
        llm = ChatOpenAI(
            temperature=0.7,  # Vyššia kreativita pre zaujímavejšie odpovede
            model_name="gpt-4o"
        )
        
        # Vytvoriť reťazec s vlastným systémovým promptom pre kreatívnejšie ale faktické odpovede
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
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
                    # Strict factual Slovak answer, always use semantic search
                    strict_prompt = (
                        prompt +
                        "\n\nOdpovedaj iba fakticky, stručne a v slovenčine. "
                        "Použi informácie, ktoré najviac významovo súvisia s otázkou, aj keď nie sú presne rovnaké. "
                        "Nepoužívaj žiadne príklady, analógie ani kreatívne rozšírenia. "
                        # "Ak nemáš dostatok informácií, povedz to jasne."
                    )
                    response = qa_chain({"question": strict_prompt, "chat_history": chat_history})
                    answer = response["answer"]
                    # Fallback if answer is empty or generic
                    fallback_phrases = [
                        "neviem", "nemám dostatok informácií", "neviem odpovedať", "nemám informácie", "I don't know", "No relevant information found"
                    ]
                    if (not answer.strip() or any(phrase in answer.lower() for phrase in fallback_phrases)) and not response["source_documents"]:
                        answer = "Nenašiel som konkrétne informácie. Skúste sa opýtať na konkrétny zákon alebo tému."
                    # Pridanie informácií o zdrojoch (unikátne)
                    if response["source_documents"]:
                        sources = set()
                        for doc in response["source_documents"]:
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source_info = doc.metadata.get('source', 'Neznámy zdroj')
                                sources.add(source_info)
                        if sources:
                            answer += "\n\n**Zdroje:**"
                            for source in sources:
                                answer += f"\n- {source}"
                    
                    # Aktualizácia placeholderu s odpoveďou
                    message_placeholder.markdown(answer)
                    
                    # Pridanie odpovede do histórie
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    # Handle encoding issues and other errors
                    error_msg = "Došlo k chybe. Prosím, skúste opäť."
                    
                    # If it's an encoding error, provide a more specific message
                    if "codec can't encode character" in str(e) or "UnicodeEncodeError" in str(e) or "UnicodeDecodeError" in str(e):
                        error_msg = "Došlo k problému s kódovaním znakov. Skúsime to vyriešiť."
                        # Try a simpler response without any special characters
                        try:
                            response = qa_chain({"question": "Summarize workplace safety laws in Slovak", "chat_history": []})
                            if response and "answer" in response:
                                message_placeholder.markdown(response["answer"])
                                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                        except Exception as inner_e:
                            print(f"Secondary error: {str(inner_e)}")
                    
                    # Show the error message
                    message_placeholder.markdown(error_msg)
                    print(f"Error detail: {str(e).__class__.__name__}: {str(e)}")
    else:
        st.error("Nepodarilo sa načítať vektorové úložisko. Skontrolujte, či sú súbory správne umiestnené v priečinku processed_data.")
else:
    # Ak nie je nastavený API kľúč, zobraziť uvítaciu správu
    with st.chat_message("assistant"):
        st.markdown("Vitajte! Pre začatie konverzácie zadajte váš OpenAI API kľúč v bočnom paneli.")
