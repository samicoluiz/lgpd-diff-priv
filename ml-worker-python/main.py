import sys
import os

# Pega o caminho absoluto da pasta onde o main.py está (ml-worker-python)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Adiciona a raiz do worker ao path para encontrar 'pipeline'
sys.path.append(current_dir)

# 2. Adiciona a pasta 'pb' ao path para o gRPC funcionar internamente
sys.path.append(os.path.join(current_dir, 'pb'))

import grpc
from concurrent import futures
from pb import privacy_pb2, privacy_pb2_grpc
from pipeline.engine import PrivacyEngine  # Agora o Python encontrará este caminho

class PrivacyService(privacy_pb2_grpc.PrivacyServiceServicer):
    def __init__(self):
        self.engine = PrivacyEngine()

    def ProcessDataset(self, request, context):
        print(f"\n--- Iniciando Processamento gRPC ---")
        print(f"[FILE] Dataset: {request.input_path}")
        print(f"[EPSILON] Recebido do Go: {request.epsilon}")
        
        # Recebendo os 4 valores do engine
        output, risk, pii_detected, utility = self.engine.run_pipeline(
            request.input_path, 
            epsilon=request.epsilon
        )

        filename_only = os.path.basename(output)
        score = float(1.0 - risk)

        print(f"[DONE] Pipeline Finalizado!")
        print(f"[METRICS] Privacidade: {score:.4f} | Utilidade (JSD): {utility:.4f}")
            
        return privacy_pb2.AnonymizeResponse(
            output_path=filename_only,
            privacy_score=score,
            utility_score=utility, # Enviando a métrica para o Dashboard
            epsilon_used=request.epsilon,
            status="Sucesso",
            pii_report={col: "MASKED" for col in pii_detected}
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    privacy_pb2_grpc.add_PrivacyServiceServicer_to_server(PrivacyService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()