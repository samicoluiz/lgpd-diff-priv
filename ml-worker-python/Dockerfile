# Utilizamos a versão 3.10-slim para garantir compatibilidade com o anonymeter
# e reduzir o tamanho final da imagem.
FROM python:3.10-slim

# Evita que o Python gere arquivos .pyc e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Instalamos dependências de sistema necessárias para compilar algumas libs de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiamos o requirements primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instalamos as bibliotecas de IA e Privacidade Diferencial
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos o código-fonte do worker
COPY . .

# Criamos o ponto de montagem para o volume compartilhado de Big Data
RUN mkdir -p /app/data

# Executamos o servidor gRPC
CMD ["python", "main.py"]