import sys
import subprocess

def install_requirements():
    print("Aggiornamento di pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"Avviso: impossibile aggiornare pip: {e}")

    requirements = [
        "numpy",
        "opencv-python",
        "pillow",
        "astropy",
        "scipy",
        "tkinterdnd2"
    ]
    
    print("\nInstallazione dei requisiti richiesti da FitsManager...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        print("\nInstallazione completata con successo!")
    except Exception as e:
        print(f"\nErrore durante l'installazione delle dipendenze: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
