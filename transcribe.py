import whisper
import torch
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Triton kernels.*")

def verificar_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"✅ GPU detectada: {gpu_name}")
        print(f"🔥 VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"⚡ CUDA version: {cuda_version}")
        return True
    else:
        print("⚠️  GPU não detectada. Usando CPU.")
        return False

def transcrever_audio_longo(caminho_audio, modelo="base", usar_gpu=True):
    gpu_disponivel = verificar_gpu() and usar_gpu
    device = "cuda" if gpu_disponivel else "cpu"
    
    print(f"Carregando modelo Whisper '{modelo}' no {device.upper()}...")
    
    if gpu_disponivel:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    model = whisper.load_model(modelo, device=device)
    
    print(f"Iniciando transcrição de: {caminho_audio}")
    if gpu_disponivel:
        print("🚀 Usando GPU - velocidade ~10x mais rápida!")
        print("⚠️  Avisos sobre Triton são normais e não afetam o desempenho")
    else:
        print("⚠️  Para áudios de 2h+, isso pode demorar 30-60 minutos...")
    
    result = model.transcribe(
        caminho_audio,
        language="pt",              
        verbose=True,
        fp16=gpu_disponivel,        
        temperature=0.0,
        beam_size=5,
        best_of=5,
        word_timestamps=False       
    )
    
    if gpu_disponivel:
        torch.cuda.empty_cache()
    
    output_dir = Path("./texto")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "texto_gerado.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Transcrição do Áudio\n\n")
        f.write(result["text"])
    
    print(f"\n✅ Transcrição salva em: {output_path}")
    print(f"📊 Texto gerado: {len(result['text'])} caracteres")
    
    return result["text"]

def main():
    print("=== TRANSCRIÇÃO DE ÁUDIO COM WHISPER ===\n")
    
    gpu_disponivel = verificar_gpu()
    
    while True:
        caminho_audio = input("Caminho do arquivo de áudio: ").strip().strip('"')
        if os.path.exists(caminho_audio):
            break
        print("❌ Arquivo não encontrado. Tente novamente.")
    
    print("\nEscolha o modelo:")
    if gpu_disponivel:
        print("🚀 GPU DETECTADA - Tempos para áudio de 2h:")
        print("1. tiny   - ~2 minutos")
        print("2. base   - ~6 minutos (recomendado)")
        print("3. small  - ~12 minutos")
        print("4. medium - ~20 minutos")
        print("5. large  - ~40 minutos (máxima qualidade)")
        print("\n⚠️  Para sua GTX 1650 (4GB), recomendo 'base' ou 'small'")
    else:
        print("🐌 CPU - Tempos para áudio de 2h:")
        print("1. tiny   - ~20 minutos")
        print("2. base   - ~60 minutos")
        print("3. small  - ~120 minutos")
        print("4. medium - ~240 minutos")
        print("5. large  - ~480 minutos")
    
    escolha = input("Escolha (1-5, padrão=2): ").strip() or "2"
    modelos = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    modelo = modelos.get(escolha, "base")
    
    if modelo == "large" and gpu_disponivel:
        print("⚠️  ATENÇÃO: Modelo 'large' pode ser lento na GTX 1650")
        continuar = input("Continuar mesmo assim? (s/N): ").strip().lower()
        if continuar != 's':
            modelo = "base"
            print("✅ Usando modelo 'base' (mais adequado para sua GPU)")
    
    try:
        transcrever_audio_longo(caminho_audio, modelo, gpu_disponivel)
        print("\n🎉 Transcrição concluída! Execute main.py para gerar o resumo.")
    except Exception as e:
        print(f"❌ Erro na transcrição: {e}")
        if "out of memory" in str(e).lower():
            print("💡 Tente um modelo menor ou feche outros programas que usam GPU")

if __name__ == "__main__":
    main()