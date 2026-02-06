import pandas as pd
import warnings
from anonymeter.evaluators import InferenceEvaluator

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # O atacante agora conhece quase tudo (Alta Exposição)
    QIDS = ['SG_PARTIDO', 'DS_GENERO', 'DS_COR_RACA', 'DS_ESTADO_CIVIL', 'SG_UF']
    SECRET = 'DS_GRAU_INSTRUCAO'
    
    print(f"\n🕵️ [AUDITORIA TSTR] Segredo: {SECRET}")
    df_real = pd.read_parquet("df_real_train.parquet").astype(str)
    
    results = []

    for eps in [10.0, 1.0, 0.1]:
        try:
            print(f"🚀 Atacando Sintético Epsilon {eps}...")
            df_syn = pd.read_parquet(f"df_syn_eps_{eps}.parquet").astype(str)
            
            # O ataque é feito no sintético, mas validado contra o real
            eval_inf = InferenceEvaluator(
                ori=df_real, 
                syn=df_syn, 
                aux_cols=QIDS, 
                secret=SECRET, 
                n_attacks=500
            )
            eval_inf.evaluate()
            risk = eval_inf.risk()
            
            print(f"   📊 Risco de Inferência: {risk.value:.4f} (IC: {risk.ci[0]:.4f} - {risk.ci[1]:.4f})")
            results.append({'Epsilon': eps, 'Risco': risk.value})
        except Exception as e:
            print(f"⚠️ Erro no ataque Eps {eps}: {e}")

    pd.DataFrame(results).to_csv("audit_privacidade_final.csv", index=False)