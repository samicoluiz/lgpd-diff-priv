import sys
import os
import pandas as pd
import numpy as np

# --- AJUSTE DE PATH ---
root_dir = os.path.dirname(os.path.abspath(__file__))
worker_dir = os.path.join(root_dir, "ml-worker-python")
if worker_dir not in sys.path:
    sys.path.insert(0, worker_dir)

from pipeline.engine import PrivacyEngine
from pipeline.wrangling_tse import apply_wrangling
from privacy_auditor import PrivacyAuditor
from ml_utility_evaluator import MLUtilityEvaluator

def safe_round(value, precision=4):
    try:
        if value is None or np.isnan(value): return 0.0
        return round(float(value), precision)
    except: return 0.0

def run_full_audit(df_ori, df_syn, aux_cols, secret_col='CD_COR_RACA'):
    """Executa a auditoria de 3 eixos (Singling-Out, Linkability, Inference)."""
    valid_aux = [c for c in aux_cols if c in df_ori.columns and c in df_syn.columns]
    auditor = PrivacyAuditor(df_ori, df_syn, valid_aux)
    
    return {
        "SinglingOut": safe_round(auditor.run_singling_out()),
        "Linkability": safe_round(auditor.run_linkability()),
        "Inference": safe_round(auditor.run_inference(secret_col=secret_col))
    }

def run_benchmark(input_path):
    TARGET_N = 100000 
    # AJUSTADO: Usando a coluna que existe no seu Wrangler (DS_...)
    ML_TARGET = 'DS_SIT_TOT_TURNO'
    
    print(f"🚀 Iniciando Auditoria e Avaliação de ML (N={TARGET_N})")
    print(f"🎯 Target do Modelo: {ML_TARGET}")
    
    engine = PrivacyEngine()
    ml_evaluator = MLUtilityEvaluator(target_col=ML_TARGET)
    
    df_full = pd.read_parquet(input_path)
    df_working = df_full.sample(n=min(len(df_full), TARGET_N), random_state=42).reset_index(drop=True)
    
    results = []

    # --- FUNÇÃO AUXILIAR DE EXECUÇÃO ---
    def execute_scenario(scenario_name, strategy, epsilon=None):
        print(f"\n[{scenario_name}] Processando...")
        
        # 1. WRANGLING
        df_proc = apply_wrangling(df_working, strategy=strategy)
        
        # PONTE DE SEGURANÇA: Garante que o target do ML sobreviva ao wrangling
        if ML_TARGET not in df_proc.columns:
            df_proc[ML_TARGET] = df_working[ML_TARGET].astype(str).str.strip().str.upper()

        # 2. SÍNTESE (Se houver epsilon)
        if epsilon:
            engine._train_model(df_proc, epsilon=epsilon)
            df_syn, _ = engine._generate_data(df_proc)
        else:
            df_syn = df_proc # Para o cenário "Only Wrangling"

        # 3. AUDITORIA DE PRIVACIDADE
        # Define colunas auxiliares baseado no cenário
        if "FAIXA_ETARIA" in df_proc.columns:
            aux = ['SG_UF', 'SG_PARTIDO', 'FAIXA_ETARIA', 'CD_GENERO']
        else:
            aux = ['SG_UF', 'SG_PARTIDO', 'CD_GENERO']
        
        audit_res = run_full_audit(df_proc, df_syn, aux)
        util_jsd, _ = engine.calculate_utility(df_proc, df_syn)
        
        # 4. UTILIDADE DE MACHINE LEARNING
        print(f"   [ML] Treinando RandomForest/XGBoost para {ML_TARGET}...")
        ml_res = ml_evaluator.run_evaluation(df_proc, df_syn)
        
        return {
            "Cenário": scenario_name, 
            "Epsilon": epsilon if epsilon else "N/A", 
            "JSD": safe_round(util_jsd), 
            **audit_res, **ml_res
        }

    # --- EXECUÇÃO DOS CENÁRIOS ---
    
    # Cenário A: Alta Fidelidade (O nosso "vilão" de privacidade)
    results.append(execute_scenario("High-Fidelity", "high_fidelity", epsilon=1.0))

    # Cenário B: Apenas Tratamento (O nosso "vilão" de utilidade)
    results.append(execute_scenario("Wrangling-Only", "intensive", epsilon=None))

    # Cenário C: O Protocolo Proposto (Híbrido)
    for eps in [0.1, 1.0, 10.0]:
        results.append(execute_scenario(f"Intensive-eps-{eps}", "intensive", epsilon=eps))

    # SALVAMENTO
    df_final = pd.DataFrame(results)
    df_final.to_csv("benchmark_final_completo.csv", index=False, sep=';', encoding='utf-8')
    
    print("\n✅ Auditoria e Avaliação de ML finalizadas!")
    print(df_final[["Cenário", "JSD", "Inference", "RandomForest_F1_Syn", "XGBoost_F1_Syn"]])

if __name__ == "__main__":
    PATH = "backend-go/data/raw_consulta_cand_2024_BRASIL.parquet"
    if os.path.exists(PATH):
        run_benchmark(PATH)