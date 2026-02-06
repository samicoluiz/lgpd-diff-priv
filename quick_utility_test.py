import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from synthcity.plugins import Plugins
import warnings

warnings.filterwarnings("ignore")

# --- 1. ENGENHARIA DE ATRIBUTOS EXPERT ---

def extract_titles(name):
    name = str(name).upper()
    # Padrões comuns de títulos que conferem autoridade/voto
    titles = ['PASTOR', 'BISPO', 'PADRE', 'IRMÃO', 'IRMÃ', 'PROFESSOR', 'PROF', 
              'DOUTOR', 'DR', 'DRA', 'CORONEL', 'COL', 'SARGENTO', 'SGT', 'DELEGADO']
    for title in titles:
        if title in name:
            return 1
    return 0

def apply_expert_engineering(df):
    d = df.copy()
    
    # Limpeza de Nulos (Placeholder do TSE)
    d = d.replace(['-1', '-3', -1, -3, '#NULO', '#NE', 'NÃO DIVULGÁVEL'], np.nan)
    
    # A) Extração de Títulos e Comprimento do Nome
    d['TEM_TITULO'] = d['NM_URNA_CANDIDATO'].apply(extract_titles)
    d['LEN_NM_URNA'] = d['NM_URNA_CANDIDATO'].astype(str).apply(len)
    
    # B) Simplificação demográfica
    d['DS_GRAU_INSTRUCAO'] = d['DS_GRAU_INSTRUCAO'].apply(
        lambda x: 'SUPERIOR' if 'SUPERIOR COMPLETO' in str(x).upper() else 'OUTROS'
    )
    d['DS_ESTADO_CIVIL'] = d['DS_ESTADO_CIVIL'].apply(
        lambda x: 'CASADO' if 'CASADO' in str(x).upper() else 'OUTROS'
    )
    
    # C) Base Natal (Concorre onde nasceu)
    d['BASE_NATAL'] = (d['SG_UF'] == d['SG_UF_NASCIMENTO']).astype(int)
    
    return d

def add_competitive_context(df_full):
    # D) DENSIDADE: Quantos candidatos concorrem por cargo na mesma cidade?
    # Isso é feito ANTES do sample para pegar a densidade real do Brasil
    counts = df_full.groupby(['SG_UE', 'CD_CARGO'])['SQ_CANDIDATO'].transform('count')
    df_full['COMPETICAO_CARGO'] = counts
    
    # E) TAMANHO DO PARTIDO: Total de candidatos do partido (Proxy de Fundo Eleitoral)
    party_size = df_full.groupby('SG_PARTIDO')['SQ_CANDIDATO'].transform('count')
    df_full['TAMANHO_PARTIDO'] = party_size
    
    return df_full

def run_ml_benchmark(df_train, df_test, target, best_params=None):
    # Simplificação do Alvo
    df_train[target] = df_train[target].apply(lambda x: 1 if 'ELEITO' in str(x).upper() else 0)
    df_test[target] = df_test[target].apply(lambda x: 1 if 'ELEITO' in str(x).upper() else 0)
    
    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train.drop(columns=[target, 'SQ_CANDIDATO', 'NM_URNA_CANDIDATO', 'SG_UF_NASCIMENTO'], errors='ignore')
    X_test = df_test.drop(columns=[target, 'SQ_CANDIDATO', 'NM_URNA_CANDIDATO', 'SG_UF_NASCIMENTO'], errors='ignore')

    # Encoding Categórico
    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    spw = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    if best_params is None:
        print("🔎 Grid Search Expert em curso...")
        param_grid = {'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1], 'n_estimators': [300]}
        xgb = XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42, tree_method='hist')
        grid = GridSearchCV(xgb, param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"✅ Melhores parâmetros: {grid.best_params_}")
        model = grid.best_estimator_
        params = grid.best_params_
    else:
        model = XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        params = best_params
    
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average='weighted'), accuracy_score(y_test, preds), params

if __name__ == "__main__":
    PATH = "backend-go/data/raw_consulta_cand_2024_BRASIL.parquet"
    COLS = ['SQ_CANDIDATO', 'NM_URNA_CANDIDATO', 'CD_CARGO', 'SG_PARTIDO', 'SG_UF', 'SG_UE',
            'DS_GRAU_INSTRUCAO', 'DS_ESTADO_CIVIL', 'SG_UF_NASCIMENTO', 'DS_SIT_TOT_TURNO']
    
    df_raw = pd.read_parquet(PATH, columns=COLS)
    
    # 1. Contexto Competitivo (No dataset todo antes da amostragem)
    print("🌍 Calculando densidade competitiva por município...")
    df_raw = add_competitive_context(df_raw)
    
    # 2. Amostragem
    df_raw = df_raw.sample(min(35000, len(df_raw)), random_state=42)
    
    TARGET = 'DS_SIT_TOT_TURNO'
    df_eng = apply_expert_engineering(df_raw)
    df_eng = df_eng.dropna(subset=[TARGET]) # Garante alvo limpo
    
    print(f"✅ Registros prontos para treino Expert: {len(df_eng)}")

    # 3. Split
    df_train, df_test = train_test_split(df_eng, test_size=0.2, random_state=42, stratify=df_eng[TARGET])
    
    print("\n" + "="*60)
    print("📊 BENCHMARK EXPERT: DENSIDADE + TÍTULOS + GRID SEARCH")
    print("="*60)

    f1_real, acc_real, opt_params = run_ml_benchmark(df_train, df_test, TARGET)
    print(f"🏆 BASELINE (REAL) EXPERT: F1: {f1_real:.4f} | ACC: {acc_real:.4f}")

    results = [{'Epsilon': 'Original', 'F1': f1_real, 'ACC': acc_real}]

    # 4. Loop de Privacidade (Geração dos Parquets)
    for eps in [0.1, 1.0, 10.0]:
        print(f"\n🚀 Sintetizando Epsilon {eps}...")
        syn_model = Plugins().get("aim", epsilon=eps)
        syn_model.fit(df_train)
        df_syn = syn_model.generate(count=len(df_train)).dataframe()
        
        f1_syn, acc_syn, _ = run_ml_benchmark(df_syn, df_test, TARGET, best_params=opt_params)
        print(f"🔹 Epsilon {eps}: F1: {f1_syn:.4f} | ACC: {acc_syn:.4f}")
        
        df_syn.to_parquet(f"df_syn_eps_{eps}.parquet")
        results.append({'Epsilon': eps, 'F1': f1_syn, 'ACC': acc_syn})

    df_train.to_parquet("df_real_auditoria.parquet")
    print("\n" + "="*60 + "\n", pd.DataFrame(results).to_string(index=False))