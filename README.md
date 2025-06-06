# Chatbot o Bezpečnosti pri práci

Tento chatbot poskytuje informácie o slovenských zákonoch týkajúcich sa bezpečnosti pri práci na základe spracovaných údajov v vektorovom úložisku.

## Požiadavky

- Python 3.7 alebo novší
- OpenAI API kľúč
- Nainštalované knižnice z requirements.txt

## Inštalácia

1. Nainštalujte potrebné knižnice:

```bash
pip install -r requirements.txt
```

## Použitie

### Lokálne spustenie

Chatbot môžete spustiť jednoducho príkazom:

```bash
python3 run.py
```

Alebo priamo cez Streamlit:

```bash
streamlit run app.py --theme.base dark
```

### Použitie aplikácie

1. Po spustení aplikácie sa otvorí webový prehliadač s chatbotom
2. Zadajte váš OpenAI API kľúč do textového poľa v bočnom paneli
3. Pýtajte sa chatbota otázky o bezpečnosti pri práci a slovenských zákonoch

## Nasadenie na Streamlit Cloud

1. Vytvorte účet na [Streamlit Cloud](https://streamlit.io/cloud)
2. Vytvorte GitHub repozitár s vašou aplikáciou
3. Nahrajte súbory app.py, requirements.txt a priečinok processed_data do repozitára
4. Na Streamlit Cloud, kliknite na "New app" a vyberte tento repozitár
5. V nastaveniach aplikácie môžete definovať tajné kľúče (secrets)

### Nastavenie tajných kľúčov v Streamlit Cloud

Vytvorte v repozitári súbor `.streamlit/secrets.toml` s obsahom:

```toml
OPENAI_API_KEY = "váš-api-kľúč"
```

*Poznámka*: Pre bezpečnosť necommitujte tento súbor do verejného repozitára. Namiesto toho ho pridajte do `.gitignore` a pridajte tajné kľúče priamo v Streamlit Cloud nastaveniach.
