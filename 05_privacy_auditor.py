import pandas as pd
import glob, re, warnings
from anonymeter.evaluators import InferenceEvaluator

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    QIDS = ['SG_PARTIDO', 'DS_GENERO', 'DS_COR_RACA', 'DS_ESTADO_CIVIL', 'SG_UF']
    SECRET = 'DS_GRAU_INSTRUCAO'
    
    df_real = pd.read_parquet("df_real_train.parquet").astype(str)
    
    files = glob.glob("df_syn_eps_*.parquet")
    eps_files = sorted([(float(re.findall(r"eps_(.*)\.parquet", f)[0]), f) for f in files], key=lambda x: x[0], reverse=True)

    print("\n" + "="*60)
    print(f"{'Epsilon':>10} | {'Risco':>12} | {'IC 95%':>25}")
    print("-" * 60)

    for eps, fname in eps_files:
        try:
            df_syn = pd.read_parquet(fname).astype(str)
            # Aumentamos para 1000 ataques para estabilizar o IC
            eval_inf = InferenceEvaluator(ori=df_real, syn=df_syn, aux_cols=QIDS, secret=SECRET, n_attacks=1000)
            eval_inf.evaluate()
            risk = eval_inf.risk()
            ic = f"({risk.ci[0]:.3f} - {risk.ci[1]:.3f})"
            print(f"{eps:10.3f} | {risk.value:12.4f} | {ic:>25}")
        except Exception as e:
            print(f"⚠️ Erro no Epsilon {eps}: {e}")
    print("="*60)