import sys
from transcribe import main as transcribe_main

def executar_pipeline():
    print("🎵 === INICIANDO PIPELINE DE TRANSCRIÇÃO (GROQ API) ===\n")
    
    try:
        # Chama a função main do script de transcrição
        transcribe_main()
    except KeyboardInterrupt:
        print("\n❌ Operação interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro crítico no pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline()