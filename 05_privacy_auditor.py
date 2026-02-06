import pandas as pd
import warnings
from anonymeter.evaluators import InferenceEvaluator

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    QIDS = ['SG_PARTIDO', 'DS_GENERO', 'DS_COR_RACA']
    SECRET = 'DS_GRAU_INSTRUCAO'
    
    df_real = pd.read_parquet("df_real_train.parquet").astype(str)
    print(f"\n🕵️ PRIVACIDADE: Risco de Inferência ({SECRET})")
    
    final_audit = []

    for eps in [10.0, 1.0, 0.1]:
        try:
            df_syn = pd.read_parquet(f"df_syn_eps_{eps}.parquet").astype(str)
            
            # 500 ataques para reduzir a margem de erro (IC)
            eval_inf = InferenceEvaluator(ori=df_real, syn=df_syn, aux_cols=QIDS, secret=SECRET, n_attacks=500)
            eval_inf.evaluate()
            risk = eval_inf.risk()
            
            print(f"🚀 Epsilon {eps} | Risco: {risk.value:.4f}")
            final_audit.append({'Epsilon': eps, 'Risco': risk.value})
        except: continue

    pd.DataFrame(final_audit).to_csv("audit_privacidade_final.csv", index=False)