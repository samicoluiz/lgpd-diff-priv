import os
import pandas as pd
import torch
import gc
from tqdm import tqdm
from pipeline.engine import PrivacyEngine
from pipeline.ml_utility_evaluator import run_ml_comparison

# ==========================================
# CONFIGURAÇÕES DO EXPERIMENTO
# ==========================================
RAW_DATA_PATH = "../backend-go/data/raw_consulta_cand_2024_BRASIL.csv"
# Adicionado 0.1 para mostrar o limite de proteção vs utilidade
EPSILONS = [0.1, 1.0, 10.0, 100.0] 
TARGET = "CD_GENERO" 

def cleanup_gpu(engine_instance):
    """Limpa referências e esvazia o cache da RTX 4080."""
    if hasattr(engine_instance, 'synth_model'):
        del engine_instance.synth_model
        engine_instance.synth_model = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("\n[MEM] VRAM liberada para a próxima rodada.")

# ==========================================
# LOOP DE EXECUÇÃO AUTOMATIZADA
# ==========================================
engine = PrivacyEngine()
summary_report = []

# Barra de progresso para acompanhar o ETA (Tempo Estimado)
pbar = tqdm(EPSILONS, desc="🧪 Experimentos TCC", unit="epsilon")

for eps in pbar:
    pbar.set_description(f"🧪 Epsilon {eps}")
    
    try:
        # 1. Geração Sintética (Pipeline Engine)
        pbar.set_postfix({"fase": "Geração AIM"})
        output_path, risk, pii_cols, util_m = engine.run_pipeline(RAW_DATA_PATH, epsilon=eps)
        
        # Validação de segurança: Só prossegue se o arquivo foi gerado
        if not output_path or not os.path.exists(output_path):
            print(f"\n[!] Erro: Falha ao gerar dado sintético para Epsilon {eps}")
            continue
            
        # 2. Avaliação de Utilidade Preditiva (ML Evaluator)
        pbar.set_postfix({"fase": "Treino ML"})
        ml_res = run_ml_comparison(
            path_real=RAW_DATA_PATH, 
            path_synth=output_path, 
            df_clean=engine.last_df_clean, 
            target_col=TARGET, 
            epsilon_label=str(eps)
        )
        
        # 3. Consolidação dos Resultados
        summary_report.append({
            "Epsilon": eps,
            "F1_Real": f"{ml_res['Real']['F1']:.4f}",
            "F1_Wrangled": f"{ml_res['Wrangled']['F1']:.4f}",
            "F1_Synthetic": f"{ml_res['Synthetic']['F1']:.4f}",
            "Privacy_Score": f"{(1.0 - risk):.4f}",
            "Utility_JSD": f"{util_m:.4f}"
        })
        
    except Exception as e:
        print(f"\n[!] Erro crítico no Epsilon {eps}: {str(e)}")
        
    finally:
        # Garante que a GPU seja limpa mesmo após erros
        cleanup_gpu(engine)

# ==========================================
# RELATÓRIO FINAL
# ==========================================
if summary_report:
    df_report = pd.DataFrame(summary_report)
    print("\n" + "="*40)
    print("      RELATÓRIO FINAL DE RESULTADOS")
    print("="*40)
    print(df_report.to_string(index=False))
    
    # Salva para uso posterior no visualizer.py
    df_report.to_csv("tcc_final_results.csv", index=False)
    print("\n[SUCCESS] Resultados salvos em 'tcc_final_results.csv'")
else:
    print("\n[!] Nenhum resultado foi coletado. Verifique os logs acima.")