import os
import pandas as pd
import torch
import gc
import json
from tqdm import tqdm
from pipeline.engine import PrivacyEngine
from pipeline.ml_utility_evaluator import run_ml_comparison
from privacy_auditor import PrivacyAuditor # Integrando o novo auditor

# ==========================================
# CONFIGURAÇÕES DO EXPERIMENTO
# ==========================================
RAW_DATA_PATH = "../backend-go/data/raw_consulta_cand_2024_BRASIL.csv"
EPSILONS = [0.1, 1.0, 10.0, 100.0] 
TARGET = "CD_GENERO" 
AUX_COLS = ['NM_UE', 'SG_PARTIDO', 'DT_NASCIMENTO', 'CD_GENERO'] # Para o Auditor

def cleanup_gpu(engine_instance):
    if hasattr(engine_instance, 'synth_model'):
        del engine_instance.synth_model
        engine_instance.synth_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n[MEM] VRAM liberada.")

# ==========================================
# LOOP DE EXECUÇÃO AUTOMATIZADA
# ==========================================
engine = PrivacyEngine()
summary_report = []

pbar = tqdm(EPSILONS, desc="🧪 Experimentos TCC", unit="epsilon")

for eps in pbar:
    pbar.set_description(f"🧪 Epsilon {eps}")
    
    try:
        # 1. GERAÇÃO (Engine atualizada retorna 5 valores)
        pbar.set_postfix({"fase": "AIM Generation"})
        output_path, df_ori, df_syn, pii_cols, util_jsd = engine.run_pipeline(
            RAW_DATA_PATH, 
            epsilon=eps
        )
        
        if df_syn is None: continue

        # 2. AUDITORIA PROFUNDA (PrivacyAuditor)
        pbar.set_postfix({"fase": "Privacy Audit"})
        auditor = PrivacyAuditor(df_ori=df_ori, df_syn=df_syn, aux_cols=AUX_COLS)
        
        risk_link = auditor.run_linkability()
        risk_inf  = auditor.run_inference(secret_col='CD_COR_RACA')
        
        # Calculamos o score de privacidade baseado no pior caso (worst-case)
        worst_risk = max(filter(None, [risk_link, risk_inf]), default=0.0)
        final_privacy_score = 1.0 - worst_risk
            
        # 3. UTILIDADE PREDITIVA (ML Evaluator)
        pbar.set_postfix({"fase": "ML Utility"})
        ml_res = run_ml_comparison(
            path_real=RAW_DATA_PATH, 
            path_synth=output_path, 
            df_clean=df_ori, # Usando o df_ori retornado pelo engine
            target_col=TARGET, 
            epsilon_label=str(eps)
        )
        
        # 4. CONSOLIDAÇÃO
        summary_report.append({
            "Epsilon": eps,
            "Privacy_Score": f"{final_privacy_score:.4f}",
            "Risk_Linkability": f"{risk_link:.4f}",
            "Risk_Inference": f"{risk_inf:.4f}",
            "Status_LGPD": auditor.get_risk_label(worst_risk),
            "Utility_JSD": f"{util_jsd:.4f}",
            "F1_Synthetic": f"{ml_res['Synthetic']['F1']:.4f}",
            "F1_Drop": f"{(ml_res['Wrangled']['F1'] - ml_res['Synthetic']['F1']):.4f}"
        })
        
    except Exception as e:
        print(f"\n[!] Erro no Epsilon {eps}: {str(e)}")
    finally:
        cleanup_gpu(engine)

# ==========================================
# RELATÓRIO E SALVAMENTO
# ==========================================
if summary_report:
    df_report = pd.DataFrame(summary_report)
    print("\n" + "="*80)
    print("                RESULTADOS CONSOLIDADOS DO TCC")
    print("="*80)
    print(df_report.to_string(index=False))
    
    df_report.to_csv("tcc_final_results.csv", index=False)
    print("\n[SUCCESS] Experimentos concluídos.")