import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def train_and_test(df_train, df_test, do_wrangling=False):
    df_tr, df_ts = df_train.copy(), df_test.copy()
    
    if do_wrangling:
        # Exemplo de wrangling: agrupar ocupações e simplificar instrução
        top_jobs = df_tr['DS_OCUPACAO'].value_counts().nlargest(10).index
        df_tr['DS_OCUPACAO'] = df_tr['DS_OCUPACAO'].apply(lambda x: x if x in top_jobs else 'OUTROS')
        df_ts['DS_OCUPACAO'] = df_ts['DS_OCUPACAO'].apply(lambda x: x if x in top_jobs else 'OUTROS')

    X_train, y_train = df_tr.drop(columns=['ALVO']), df_tr['ALVO']
    X_test, y_test = df_ts.drop(columns=['ALVO']), df_ts['ALVO']

    # Label Encoding (Sincronizado)
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
    real_test = pd.read_parquet("df_real_test.parquet") # O Gabarito

    print("\n🏁 INICIANDO BENCHMARK DE METODOLOGIA")
    
    # 1. Baseline: Real -> Real
    f1_base = train_and_test(real_train, real_test, do_wrangling=False)
    f1_base_wr = train_and_test(real_train, real_test, do_wrangling=True)
    
    print(f"🟢 Real Puro: {f1_base:.4f} | Real + Wrangling: {f1_base_wr:.4f}")

    for eps in [10.0, 1.0, 0.1]:
        try:
            syn_train = pd.read_parquet(f"df_syn_eps_{eps}.parquet")
            f1_syn = train_and_test(syn_train, real_test, do_wrangling=False)
            f1_syn_wr = train_and_test(syn_train, real_test, do_wrangling=True)
            
            print(f"🚀 EPS {eps} | Syn Puro: {f1_syn:.4f} | Syn + Wrangling: {f1_syn_wr:.4f}")
        except: continue