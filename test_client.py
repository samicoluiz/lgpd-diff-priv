import grpc
import sys
import os

# 1. Configurar caminhos ANTES de importar os módulos gerados
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona a pasta pb do worker ao path para o cliente encontrar os arquivos
sys.path.append(os.path.join(current_dir, 'ml-worker-python', 'pb'))

# 2. Agora sim, os imports
import privacy_pb2
import privacy_pb2_grpc

def run_test():
    # Caminho absoluto para evitar erro de "File Not Found" no Worker
    base_dir = "/mnt/c/Users/looui/Documents/projetos/tcc/lgpd-diff-priv"
    path_ao_arquivo = os.path.join(base_dir, "data/raw_test.parquet")

    # Conecta no servidor local
    channel = grpc.insecure_channel('localhost:50051')
    stub = privacy_pb2_grpc.PrivacyServiceStub(channel)

    print(f"[INFO] Enviando requisição para o Worker...")
    print(f"[FILE] Alvo: {path_ao_arquivo}")

    stub = worker_pb2_grpc.MLWorkerStub(channel)
    
    try:
        response = stub.ProcessData(
            worker_pb2.ProcessRequest(input_path=path_ao_arquivo)
        )
        print(f"[OK] Status: {response.status}")
        print(f"[SCORE] Score de Privacidade: {response.privacy_score:.4f}")
        print(f"[SAVE] Arquivo Sintético Gerado: {response.output_path}")
        print(f"[REPORT] Relatório PII: {response.pii_report}")
        
    except grpc.RpcError as e:
        print(f"[ERROR] Erro no gRPC: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run_test()