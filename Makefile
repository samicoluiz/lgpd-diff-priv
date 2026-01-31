# Variáveis
# Forçamos o uso do binário 3.10 instalado no seu WSL
PYTHON_BIN=python3.10
PROTO_SRC=api/privacy.proto
GO_OUT=backend-go/pb
PY_OUT=ml-worker-python/pb

.PHONY: all gen-proto setup-venv help

all: help

## gen-proto: Gera o código gRPC para Go e Python usando o ambiente virtual
gen-proto:
	@echo "Gerando código a partir do $(PROTO_SRC)..."
	# Gerar Go
	protoc --go_out=$(GO_OUT) --go_opt=paths=source_relative \
	       --go-grpc_out=$(GO_OUT) --go-grpc_opt=paths=source_relative \
	       -I api $(PROTO_SRC)
	# Gerar Python usando o binário de dentro do venv para garantir a versão 3.10
	./venv/bin/python -m grpc_tools.protoc -I api --python_out=$(PY_OUT) \
	       --grpc_python_out=$(PY_OUT) $(PROTO_SRC)
	@echo "Código gerado com sucesso!"

## setup-venv: Cria o venv com Python 3.10 e instala dependências de IA
setup-venv:
	@echo "Criando ambiente virtual com $(PYTHON_BIN)..."
	$(PYTHON_BIN) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install grpcio grpcio-tools synthcity presidio-analyzer \
	                     presidio-anonymizer anonymeter faker pandas pyarrow
	@echo "Ambiente Python 3.10 configurado com sucesso!"

## help: Mostra os comandos disponíveis
help:
	@echo "Comandos disponíveis:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' |  sed -e 's/^/ /'