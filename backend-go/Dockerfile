# Estágio 1: Compilação
FROM golang:1.22-alpine AS builder

# Instalação de dependências para o DuckDB (CGO pode ser necessário)
RUN apk add --no-cache build-base

WORKDIR /app

# Copia os arquivos de módulos primeiro para cache de dependências
COPY go.mod go.sum ./
RUN go mod download

# Copia o código fonte (incluindo o código gerado pelo gRPC)
COPY . .

# Compila o binário estático
# CGO_ENABLED=1 é necessário se você estiver usando o driver C do DuckDB
RUN CGO_ENABLED=1 GOOS=linux go build -o main ./cmd/main.go

# Estágio 2: Execução (Imagem Final)
FROM alpine:latest

# Instalação de bibliotecas dinâmicas mínimas para o DuckDB
RUN apk add --no-cache libstdc++ gcompat

WORKDIR /app

# Copia apenas o binário compilado do estágio anterior
COPY --from=builder /app/main .

# Garante que a pasta de dados compartilhada existe
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["./main"]