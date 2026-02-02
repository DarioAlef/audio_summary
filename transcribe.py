import os
import time
import io
import ffmpeg
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# Carrega variáveis de ambiente
load_dotenv()

# Configuração
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AUDIO_FOLDER = "audios"
OUTPUT_FILE = "texto/texto_gerado.md"
CHUNK_DURATION = 120  # 2 minutos em segundos

def get_audio_duration(file_path):
    """Retorna a duração do áudio em segundos."""
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Erro ao ler duração do arquivo {file_path}: {e.stderr.decode() if e.stderr else e}")
        raise

def extract_audio_chunk(file_path, start_time, duration):
    """
    Extrai um pedaço do áudio e retorna como um arquivo em memória (BytesIO/WAV).
    """
    try:
        out_file = io.BytesIO()
        process = (
            ffmpeg.input(file_path, ss=start_time, t=duration)
            .output("pipe:1", format="wav")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        output, err = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {err.decode() if err else 'Unknown error'}")

        out_file.write(output)
        out_file.seek(0)
        out_file.name = "chunk.wav" # Necessário para a API da Groq
        return out_file
    except Exception as e:
        print(f"Erro ao extrair chunk de {start_time}s: {str(e)}")
        raise

def transcribe_chunk(client, audio_file_obj):
    """Envia o chunk para a API da Groq e retorna o texto."""
    try:
        transcription = client.audio.transcriptions.create(
            file=audio_file_obj,
            model="whisper-large-v3-turbo",
            response_format="json",
            language="pt",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        print(f"Erro na API da Groq: {str(e)}")
        return "[ERRO NA TRANSCRIÇÃO DESTE TRECHO]"

def save_text(text, file_path):
    """Salva texto no arquivo de saída (append)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")

def process_file(file_path, client):
    """Processa um único arquivo de áudio."""
    print(f"\n🎧 Processando arquivo: {file_path}")
    duration = get_audio_duration(file_path)
    print(f"⏱️ Duração total: {duration:.2f} segundos ({duration/60:.2f} min)")
    
    # Adiciona cabeçalho ao markdown
    save_text(f"\n## Transcrição: {os.path.basename(file_path)}\n", OUTPUT_FILE)
    
    current_time = 0
    chunk_count = 1
    
    while current_time < duration:
        print(f"  🔄 Processando parte {chunk_count}: {current_time/60:.2f}m - {(current_time + CHUNK_DURATION)/60:.2f}m")
        
        # Extrai chunk
        audio_chunk = extract_audio_chunk(file_path, current_time, CHUNK_DURATION)
        
        # Transcreve
        text = transcribe_chunk(client, audio_chunk)
        print(f"    ✅ Transcrição recebida ({len(text)} chars)")
        
        # Salva
        save_text(text, OUTPUT_FILE)
        print(f"    💾 Salvo em {OUTPUT_FILE}")
        
        current_time += CHUNK_DURATION
        chunk_count += 1
        
        # Pequena pausa para não abusar do rate limit
        time.sleep(1)

def main():
    if not GROQ_API_KEY:
        print("❌ Erro: GROQ_API_KEY não encontrada no arquivo .env")
        return

    client = Groq(api_key=GROQ_API_KEY)
    
    # Cria diretório de audios se não existir
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
        print(f"📁 Pasta '{AUDIO_FOLDER}' criada. Coloque seus arquivos nela.")
        return

    # Lista arquivos de áudio
    audio_extensions = ('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus')
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(audio_extensions)]
    
    if not files:
        print(f"⚠️  Nenhum arquivo de áudio encontrado em '{AUDIO_FOLDER}'")
        return

    print(f"🚀 Iniciando transcrição de {len(files)} arquivos...")
    
    # Limpa ou cria arquivo de saída (opcional: aqui estou dando append, 
    # mas se for rodar do zero talvez o usuário queira limpar. 
    # O user disse 'será salvo incrementalmente', não disse limpar.
    # Vou manter append, mas colocar um separador de início de execução.)
    save_text(f"\n\n---\n# Nova Execução: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", OUTPUT_FILE)

    for file_name in files:
        process_file(os.path.join(AUDIO_FOLDER, file_name), client)
    
    print("\n🎉 Processamento concluído!")

if __name__ == "__main__":
    main()