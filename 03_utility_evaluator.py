import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def run_benchmark(df_train, df_test):
    # Sincronia: Features que o atacante conhece vs Alvo que ele quer descobrir
    features = ['SG_PARTIDO', 'DS_GENERO', 'DS_COR_RACA', 'DS_OCUPACAO']
    target = 'DS_GRAU_INSTRUCAO'
    
    X_train, y_train = df_train[features], df_train[target].astype(str)
    X_test, y_test = df_test[features], df_test[target].astype(str)
    
    # Encoding para features e alvo (Instrução é categórica)
    le_target = LabelEncoder()
    le_target.fit(pd.concat([y_train, y_test]))
    y_train = le_target.transform(y_train)
    y_test = le_target.transform(y_test)

    for col in X_train.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
            
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # F1-weighted é ideal para classes desbalanceadas (ex: Ensino Superior vs Primário)
    return f1_score(y_test, model.predict(X_test), average='weighted')

if __name__ == "__main__":
    print("\n📊 UTILIDADE: Preditividade do Grau de Instrução")
    df_test = pd.read_parquet("df_real_test.parquet")
    f1_orig = run_benchmark(pd.read_parquet("df_real_train.parquet"), df_test)
    
    results = [{'Epsilon': 'Original', 'F1': round(f1_orig, 4), 'Retenção': '100%'}]
    
    for eps in [10.0, 1.0, 0.1]:
        try:
            f1_syn = run_benchmark(pd.read_parquet(f"df_syn_eps_{eps}.parquet"), df_test)
            ret = (f1_syn / f1_orig) * 100
            results.append({'Epsilon': eps, 'F1': round(f1_syn, 4), 'Retenção': f'{ret:.2f}%'})
        except: continue

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv("benchmark_utilidade_final.csv", index=False)