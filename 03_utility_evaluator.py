import pandas as pd
import glob, re, warnings
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def apply_wrangling(df):
    """Tratamento manual para tentar recuperar utilidade (Wrangling)"""
    df_w = df.copy()
    # Agrupamento Top 15 (Equilíbrio entre sinal e ruído)
    for col in ['DS_OCUPACAO', 'SG_PARTIDO']:
        top_n = df_w[col].value_counts().nlargest(15).index
        df_w[col] = df_w[col].apply(lambda x: x if x in top_n else 'OUTROS')
    return df_w

def run_model(df_train, df_test, wrangle=False):
    df_tr = apply_wrangling(df_train) if wrangle else df_train.copy()
    df_ts = df_test.copy() 

    X_train, y_train = df_tr.drop(columns=['ALVO']), df_tr['ALVO']
    X_test, y_test = df_ts.drop(columns=['ALVO']), df_ts['ALVO']

    for col in X_train.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test), average='weighted')

if __name__ == "__main__":
    real_train = pd.read_parquet("df_real_train.parquet")
    real_test = pd.read_parquet("df_real_test.parquet")

    # Baselines Reais
    f1_real_puro = run_model(real_train, real_test, wrangle=False)
    f1_real_wrang = run_model(real_train, real_test, wrangle=True)

    print("\n" + "="*75)
    print(f"{'Epsilon':>10} | {'F1 (Puro)':>12} | {'F1 (Wrangled)':>15} | {'Retenção %':>12}")
    print("-" * 75)
    print(f"{'REAL':>10} | {f1_real_puro:12.4f} | {f1_real_wrang:15.4f} | {'100.00%':>12}")
    print("-" * 75)

    files = glob.glob("df_syn_eps_*.parquet")
    eps_files = sorted([(float(re.findall(r"eps_(.*)\.parquet", f)[0]), f) for f in files], key=lambda x: x[0], reverse=True)

    for eps, fname in eps_files:
        syn_train = pd.read_parquet(fname)
        f1_p = run_model(syn_train, real_test, wrangle=False)
        f1_w = run_model(syn_train, real_test, wrangle=True)
        # Retenção baseada no Real Wrangled
        ret = (f1_w / f1_real_wrang) * 100
        print(f"{eps:10.3f} | {f1_p:12.4f} | {f1_w:15.4f} | {ret:11.2f}%")
    print("="*75)