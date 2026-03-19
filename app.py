import subprocess
import time
import sys
import os

def run_api():
    """Lancer FastAPI avec uvicorn en sous-processus"""
    cmd = [
        "uvicorn",
        "api:app",           # ton fichier main.py
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    print("Démarrage de l'API FastAPI...")
    subprocess.Popen(cmd)

def run_streamlit():
    """Lancer Streamlit dans un sous-processus"""
    cmd = ["streamlit", "run", "interface.py"]
    print("Démarrage de Streamlit...")
    subprocess.run(cmd)

if __name__ == "__main__":
    # Démarrer l'API
    run_api()
    
    # Attendre quelques secondes que l'API soit prête
    time.sleep(3)
    
    # Démarrer Streamlit
    run_streamlit()