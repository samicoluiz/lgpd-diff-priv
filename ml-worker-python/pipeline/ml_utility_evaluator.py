import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run_ml_comparison(path_real, path_synth, df_clean, target_col, epsilon_label):
    print(f"\n Avaliando Utilidade Preditiva [Target: {target_col} | Epsilon: {epsilon_label}]")
    os.makedirs("models/evaluators", exist_index=True)
    
    # 1. Carregamento
    df_real = pd.read_csv(path_real, sep=';', encoding='iso-8859-1', low_memory=False).sample(100000, random_state=42)
    df_synth = pd.read_parquet(path_synth)
    df_wrangled = df_clean.copy() # O dado que veio do wrangling no pipeline

    # 2. Pré-processamento (Label Encoding Uniforme)
    combined = pd.concat([df_real, df_synth, df_wrangled], axis=0).astype(str)
    for col in df_real.columns:
        le = LabelEncoder()
        le.fit(combined[col])
        df_real[col] = le.transform(df_real[col].astype(str))
        df_synth[col] = le.transform(df_synth[col].astype(str))
        df_wrangled[col] = le.transform(df_wrangled[col].astype(str))

    # 3. Preparação (Split Real para Teste Final)
    X_r = df_real.drop(columns=[target_col])
    y_r = df_real[target_col]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    # 4. Modelos
    models = {
        "Real": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Wrangled": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Synthetic": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, clf in models.items():
        print(f" - Treinando modelo {name}...")
        if name == "Real":
            clf.fit(X_train_r, y_train_r)
        elif name == "Wrangled":
            clf.fit(df_wrangled.drop(columns=[target_col]), df_wrangled[target_col])
        else:
            clf.fit(df_synth.drop(columns=[target_col]), df_synth[target_col])
        
        # Salva o modelo para sua auditoria no TCC
        joblib.dump(clf, f"models/evaluators/rf_{name.lower()}_eps_{epsilon_label}.joblib")
        
        # Predição sempre no teste REAL
        y_pred = clf.predict(X_test_r)
        results[name] = {
            "Acc": accuracy_score(y_test_r, y_pred),
            "Prec": precision_score(y_test_r, y_pred, average='weighted'),
            "F1": f1_score(y_test_r, y_pred, average='weighted')
        }

    return results