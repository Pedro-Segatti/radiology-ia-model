# Usando a imagem oficial do Python como base
FROM python:3.9-slim

# Definindo o diretório de trabalho dentro do contêiner para a pasta src
WORKDIR /app

# Copiar o arquivo de requisitos para dentro do contêiner
COPY requirements.txt /app

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação para o contêiner
COPY . /app

# Definir variáveis de ambiente
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=1
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

# Expor a porta em que a aplicação vai rodar
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["flask", "run"]
