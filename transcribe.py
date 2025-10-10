import whisper
import torch
import os
import warnings
from pathlib import Path
import math

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

def transcrever_audio_segmentado(caminho_audio, modelo="base", usar_gpu=True, tamanho_segmento=1800):
    """
    Transcreve áudio longo dividindo em segmentos para evitar repetições.
    tamanho_segmento: duração em segundos (padrão: 30 minutos)
    """
    gpu_disponivel = verificar_gpu() and usar_gpu
    device = "cuda" if gpu_disponivel else "cpu"
    
    print(f"Carregando modelo Whisper '{modelo}' no {device.upper()}...")
    
    if gpu_disponivel:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    model = whisper.load_model(modelo, device=device)
    
    print(f"🔄 Processando áudio em segmentos de {tamanho_segmento//60} minutos para evitar repetições...")
    
    # Carrega o áudio completo para obter informações
    audio = whisper.load_audio(caminho_audio)
    duracao_total = len(audio) / whisper.audio.SAMPLE_RATE
    
    print(f"📊 Duração total do áudio: {duracao_total/60:.1f} minutos")
    
    # Calcula número de segmentos
    num_segmentos = math.ceil(duracao_total / tamanho_segmento)
    print(f"🔢 Dividindo em {num_segmentos} segmentos")
    
    texto_completo = ""
    
    for i in range(num_segmentos):
        inicio = i * tamanho_segmento * whisper.audio.SAMPLE_RATE
        fim = min((i + 1) * tamanho_segmento * whisper.audio.SAMPLE_RATE, len(audio))
        
        segmento_audio = audio[int(inicio):int(fim)]
        
        print(f"\n🎯 Processando segmento {i+1}/{num_segmentos} ({inicio/whisper.audio.SAMPLE_RATE/60:.1f}-{fim/whisper.audio.SAMPLE_RATE/60:.1f} min)")
        
        # Configurações anti-repetição para cada segmento
        result = model.transcribe(
            segmento_audio,
            language="pt",
            verbose=False,  # Reduz spam no console
            fp16=gpu_disponivel,
            temperature=0.2,
            beam_size=1,
            best_of=1,
            word_timestamps=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            hallucination_silence_threshold=3.0
        )
        
        # Adiciona uma quebra entre segmentos
        if i > 0:
            texto_completo += "\n\n"
        texto_completo += result["text"]
        
        print(f"✅ Segmento {i+1} concluído: {len(result['text'])} caracteres")
        
        # Limpa cache da GPU entre segmentos
        if gpu_disponivel:
            torch.cuda.empty_cache()
    
    return texto_completo

def transcrever_audio_longo(caminho_audio, modelo="base", usar_gpu=True, usar_segmentacao=True):
    """
    Função principal de transcrição com opção de segmentação automática.
    usar_segmentacao: Se True, divide áudios longos em segmentos (recomendado)
    """
    if usar_segmentacao:
        # Usa a nova função segmentada para evitar repetições
        texto = transcrever_audio_segmentado(caminho_audio, modelo, usar_gpu)
    else:
        # Função original (pode ter problemas com áudios longos)
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
        
        # Configurações para evitar repetições em áudios longos
        result = model.transcribe(
            caminho_audio,
            language="pt",              
            verbose=True,
            fp16=gpu_disponivel,        
            temperature=0.2,            # Aumenta diversidade, evita repetições
            beam_size=1,                # Reduz para evitar loops
            best_of=1,                  # Reduz para evitar loops
            word_timestamps=False,
            no_speech_threshold=0.6,    # Detecta melhor silêncios
            logprob_threshold=-1.0,     # Filtra tokens com baixa confiança
            compression_ratio_threshold=2.4,  # Detecta repetições
            condition_on_previous_text=False,  # Evita dependência de texto anterior
            hallucination_silence_threshold=3.0  # Detecta alucinações em silêncios
        )
        
        if gpu_disponivel:
            torch.cuda.empty_cache()
        
        texto = result["text"]
    
    # Salva o resultado
    output_dir = Path("./texto")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "texto_gerado.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Transcrição do Áudio\n\n")
        f.write(texto)
    
    print(f"\n✅ Transcrição salva em: {output_path}")
    print(f"📊 Texto gerado: {len(texto)} caracteres")
    
    return texto

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
    
    # Nova opção para escolher método de transcrição
    print("\n🎯 Método de transcrição:")
    print("1. Segmentado (recomendado para áudios >30min) - Evita repetições")
    print("2. Completo (mais rápido, mas pode repetir em áudios longos)")
    
    metodo = input("Escolha (1-2, padrão=1): ").strip() or "1"
    usar_segmentacao = metodo == "1"
    
    if usar_segmentacao:
        print("✅ Usando método segmentado - divide o áudio para evitar repetições")
    else:
        print("⚠️  Método completo - pode repetir frases em áudios muito longos")
    
    try:
        transcrever_audio_longo(caminho_audio, modelo, gpu_disponivel, usar_segmentacao)
        print("\n🎉 Transcrição concluída! Execute summary_text.py para gerar o resumo.")
    except Exception as e:
        print(f"❌ Erro na transcrição: {e}")
        if "out of memory" in str(e).lower():
            print("💡 Tente um modelo menor ou feche outros programas que usam GPU")

if __name__ == "__main__":
    main()