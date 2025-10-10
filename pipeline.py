import subprocess
import sys
import os

def executar_pipeline():
    print("🎵 === PIPELINE COMPLETO: ÁUDIO → RESUMO ===\n")
    
    # Etapa 1: Transcrição
    print("📝 ETAPA 1: Transcrição do áudio")
    print("-" * 40)
    result = subprocess.run([sys.executable, "transcribe.py"], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print("❌ Erro na transcrição. Pipeline interrompido.")
        return
    
    # Verificar se arquivo foi gerado
    if not os.path.exists("./texto/texto_gerado.md"):
        print("❌ Arquivo de transcrição não foi gerado.")
        return
    
    # Etapa 2: Resumo
    print("\n🤖 ETAPA 2: Geração do resumo")
    print("-" * 40)
    result = subprocess.run([sys.executable, "main.py"], 
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("📄 Arquivos gerados:")
        print("   - ./texto/texto_gerado.md (transcrição)")
        print("   - ./texto/resumo_final.md (resumo)")
    else:
        print("❌ Erro na geração do resumo.")

if __name__ == "__main__":
    executar_pipeline()