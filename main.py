import os
from pathlib import Path
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

def main():
    print("=== RESUMO DE TRANSCRIÇÃO ===\n")
    
    # Verificar se arquivo existe
    texto_path = "./texto/texto_gerado.md"
    if not os.path.exists(texto_path):
        print("❌ Arquivo texto_gerado.md não encontrado!")
        print("Execute transcribe.py primeiro para gerar a transcrição.")
        return
    
    # Verificar se arquivo não está vazio
    if os.path.getsize(texto_path) == 0:
        print("❌ Arquivo texto_gerado.md está vazio!")
        return
    
    print("✅ Arquivo de transcrição encontrado.")
    
    # Configurar modelo
    model_path = "./model/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print("📥 Carregando modelo LLM...")
    llm = CTransformers(
        model=model_path,
        model_type="mistral",
        config={
            'max_new_tokens': 1024,    # Aumentado para resumos melhores
            'context_length': 4096,
            'temperature': 0.1,
            'threads': os.cpu_count()  # Usar todos os cores
        }
    )
    
    # Carregar e processar texto
    print("📖 Carregando transcrição...")
    loader = TextLoader(texto_path, encoding='utf8')
    documentos = loader.load()
    
    # Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documentos)
    
    print(f"📝 Total de chunks: {len(chunks)}")
    
    # Prompts otimizados
    MAP_PROMPT_TEMPLATE = PromptTemplate(
        template="""Analise este trecho da transcrição e extraia os pontos principais:

{text}

Pontos principais deste trecho:""",
        input_variables=["text"]
    )
    
    COMBINE_PROMPT_TEMPLATE = PromptTemplate(
        template="""Com base nos resumos dos trechos abaixo, crie um resumo final estruturado:

{text}

Crie um resumo final com:
## 📋 Resumo Executivo
[Resumo geral em 2-3 parágrafos]

## 🔑 Pontos-Chave
1. [Ponto 1]
2. [Ponto 2]
3. [Ponto 3]
4. [Ponto 4]
5. [Ponto 5]

## 🎯 Conclusões e Ações
[Principais conclusões e próximos passos]

Resumo final:""",
        input_variables=["text"]
    )
    
    # Criar cadeia de resumo
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT_TEMPLATE,
        combine_prompt=COMBINE_PROMPT_TEMPLATE,
        verbose=True
    )
    
    # Executar resumo
    print("🤖 Gerando resumo...")
    resumo_final = chain.run(chunks)
    
    # Salvar resultado
    output_path = "./texto/resumo_final.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Resumo da Transcrição\n\n")
        f.write(resumo_final)
    
    print(f"\n✅ Resumo salvo em: {output_path}")
    print("\n" + "="*50)
    print("RESUMO FINAL")
    print("="*50)
    print(resumo_final)

if __name__ == "__main__":
    main()