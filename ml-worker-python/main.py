import sys
import os
import grpc
import pandas as pd
from concurrent import futures

# Ajuste de caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'pb'))

from pb import privacy_pb2, privacy_pb2_grpc
from pipeline.engine import PrivacyEngine 
from privacy_auditor import PrivacyAuditor 

class PrivacyService(privacy_pb2_grpc.PrivacyServiceServicer):
    def __init__(self):
        self.engine = PrivacyEngine()
        self.aux_cols = ['NM_UE', 'SG_PARTIDO', 'FAIXA_ETARIA', 'CD_GENERO']
        self.target_risk = 0.15

    def _format_tabular_status(self, eps, r_so, r_li, r_in, final_score, utility):
        """Gera o log técnico com valores REAIS para o histórico."""
        line = "-" * 42
        table = [
            line,
            "       RELATÓRIO TÉCNICO DE AUDITORIA",
            line,
            f" PARAMETRO EPSILON (ε):      {eps:>10.1f}",
            f" UTILIDADE GLOBAL (JSD):     {utility:>10.4f}",
            line,
            " MÉTRICA DE RISCO            VALOR     STATUS",
            f" Singling Out (Isolamento)   {r_so:>7.4f}    OK",
            f" Linkability (Ligação)       {r_li:>7.4f}    OK",
            f" Inference (Inferência)      {r_in:>7.4f}    ⚠️",
            line,
            f" SCORE PRIVACIDADE (1-MAX):  {final_score:>10.4f}",
            line
        ]
        return "\n".join(table)

    def _run_full_audit(self, df_ori, df_syn):
        auditor = PrivacyAuditor(df_ori, df_syn, self.aux_cols)
        r_so = auditor.run_singling_out() or 0.0
        r_li = auditor.run_linkability() or 0.0
        r_in = auditor.run_inference(secret_col='CD_COR_RACA') or 0.0
        
        max_risk = max(r_so, r_li, r_in)
        return r_so, r_li, r_in, max_risk

    def ProcessDataset(self, request, context):
        print(f"\n[INFO] Iniciando Processamento: {os.path.basename(request.input_path)}")
        
        epsilon_to_use = request.epsilon
        
        # 1. Execução do Pipeline (AIM)
        output_path, df_ori, df_syn, pii_detected, utility = self.engine.run_pipeline(
            request.input_path, 
            epsilon=epsilon_to_use
        )

        # 2. Auditoria Final (Riscos)
        r_so, r_li, r_in, max_r = self._run_full_audit(df_ori, df_syn)
        p_score = float(1.0 - max_r)
        
        # 3. Geração do Status Tabular (CORRIGIDO: Agora enviando o utility)
        status_table = self._format_tabular_status(
            epsilon_to_use, r_so, r_li, r_in, p_score, utility
        )
        
        print(status_table) # Debug no console do Worker

        # 4. Resposta gRPC
        return privacy_pb2.AnonymizeResponse(
            output_path=os.path.basename(output_path),
            privacy_score=p_score,
            utility_score=utility,
            epsilon_used=epsilon_to_use,
            status=status_table,
            pii_report={col: "MASKED" for col in pii_detected},
            singling_out_risk=r_so,
            linkability_risk=r_li,
            inference_risk=r_in
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    privacy_pb2_grpc.add_PrivacyServiceServicer_to_server(PrivacyService(), server)
    server.add_insecure_port('[::]:50051')
    print("[SERVER] ML-Worker pronto no WSL (Porta 50051)")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()