### O que é

Este projeto tenho o obejtivo de conseguir transcrever grande áudios, com mais de 2h de duração.

### Como faz

Estou utilizando o modelo whisper rodando localmente via pytorch cuda 12. Rodo o arquivo main.py, escolho o modelo whisper, coloco o path do arquivo de áudio, e aguardo a transcrição. Essa transcrição fica salva em texto/texto_gerado.md

### Como rodar

1. Crie a pasta virtual
2. Instale as dependências do requirements.txt
3. Instale o pytorch de acordo com o seu Cuda instalado
4. Rode o arquivo main.py
