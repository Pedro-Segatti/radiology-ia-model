# Usando a imagem oficial do Python
FROM python:3.11-slim

# Definindo o diretório de trabalho dentro do container
WORKDIR /app

# Copiando arquivos necessários
COPY requirements.txt .
COPY src/ ./src/
COPY config.yaml .

# Instalando dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expondo a porta da API Flask
EXPOSE 5000

# Comando para rodar a API
CMD ["python3", "-m", "flask", "--app", "src/app:app", "run", "--host=0.0.0.0"]
