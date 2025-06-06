import subprocess
import os
import sys
import pkg_resources

def check_requirements():
    """Kontroluje, či sú nainštalované všetky potrebné knižnice."""
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("Súbor requirements.txt nebol nájdený.")
        return False
    
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing = []
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
        except:
            missing.append(requirement)
    
    if missing:
        print("Chýbajúce knižnice:", ", ".join(missing))
        print("Prosím, nainštalujte chýbajúce knižnice pomocou:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    # Zistenie aktuálneho adresára
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Kontrola, či existujú potrebné súbory
    vectorstore_dir = os.path.join(current_dir, "processed_data")
    if not os.path.exists(vectorstore_dir):
        print("Chyba: Priečinok processed_data nebol nájdený.")
        sys.exit(1)
    
    # Kontrola existencie index súborov
    if not (os.path.exists(os.path.join(vectorstore_dir, "index.faiss")) and 
            os.path.exists(os.path.join(vectorstore_dir, "index.pkl"))):
        print("Chyba: Súbory index.faiss alebo index.pkl neboli nájdené v priečinku processed_data.")
        sys.exit(1)
    
    # Kontrola inštalovaných knižníc
    if not check_requirements():
        print("Upozornenie: Niektoré knižnice môžu chýbať. Aplikácia nemusí fungovať správne.")
        response = input("Chcete pokračovať? (a/N): ")
        if not response.lower().startswith('a'):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("Spúšťam chatbot aplikáciu o bezpečnosti pri práci...")
    print("="*60)
    
    print("\nDôležité informácie:")
    print("Aplikácia vyžaduje OpenAI API kľúč, ktorý musíte zadať v aplikácii.")
    print("\nPre ukončenie aplikácie stlačte Ctrl+C")
    print("="*60 + "\n")
    
    # Spustenie Streamlit aplikácie
    try:
        subprocess.run(["streamlit", "run", os.path.join(current_dir, "app.py"), 
                       "--theme.base", "dark",
                       "--server.headless", "true"])
    except KeyboardInterrupt:
        print("\nAplikácia bola ukončená.")
    except Exception as e:
        print(f"Chyba pri spustení aplikácie: {e}")
        print("\nSkontrolujte, či máte nainštalované všetky potrebné knižnice:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
