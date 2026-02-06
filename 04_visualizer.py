import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def get_trained_model(df_train, target, params):
    # Prepara dados para um treino rápido focado em extrair importância
    d = df_train.copy()
    d[target] = d[target].apply(lambda x: 1 if 'ELEITO' in str(x).upper() else 0)
    y = d[target]
    X = d.drop(columns=[target, 'SQ_CANDIDATO', 'NM_URNA_CANDIDATO', 'SG_UF_NASCIMENTO'], errors='ignore')
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    model = XGBClassifier(**params, random_state=42)
    model.fit(X, y)
    return model, X.columns

def export_to_latex_data(series, filename):
    """
    Converte uma Series do Pandas em um arquivo formatado para PGFPlots.
    """
    df_tex = series.reset_index()
    df_tex.columns = ['attribute', 'importance']
    # Remove caracteres especiais que o LaTeX não gosta em nomes de colunas
    df_tex['attribute'] = df_tex['attribute'].str.replace('_', '\\_')
    df_tex.to_csv(filename, sep=' ', index=False)
    print(f"✅ Dados para LaTeX salvos em: {filename}")

if __name__ == "__main__":
    TARGET = 'DS_SIT_TOT_TURNO'
    # Parâmetros que o Grid Search encontrou no passo 03 (ajuste conforme seu resultado)
    params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
    
    df_real = pd.read_parquet("df_real_train.parquet")
    epsilons = [0.1, 1.0, 10.0]
    
    print("🎨 Gerando visualizações para o TCC...")

    # --- 1. COMPARATIVO DE IMPORTÂNCIA DE ATRIBUTOS ---
    plt.figure(figsize=(12, 8))
    
    # Modelo Real
    model_r, cols = get_trained_model(df_real, TARGET, params)
    feat_imp_r = pd.Series(model_r.feature_importances_, index=cols).sort_values(ascending=False).head(10)
    
    plt.subplot(2, 1, 1)
    sns.barplot(x=feat_imp_r.values, y=feat_imp_r.index, palette='Blues_r')
    plt.title("Top 10 Atributos Mais Importantes (Dado Real)")

    # Modelo Sintético (Epsilon 10.0)
    df_syn = pd.read_parquet("df_syn_eps_10.0.parquet")
    model_s, cols = get_trained_model(df_syn, TARGET, params)
    feat_imp_s = pd.Series(model_s.feature_importances_, index=cols).sort_values(ascending=False).head(10)

    export_to_latex_data(feat_imp_r, "feat_imp_real.dat")
    export_to_latex_data(feat_imp_s, "feat_imp_syn.dat")

    plt.subplot(2, 1, 2)
    sns.barplot(x=feat_imp_s.values, y=feat_imp_s.index, palette='Reds_r')
    plt.title("Top 10 Atributos Mais Importantes (Sintético Epsilon 10.0)")
    
    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png")
    print("✅ Gráfico de importância de atributos salvo.")

    # --- 2. MATRIZ DE CONFUSÃO (Real vs Melhor Sintético) ---
    # Para este passo, precisaríamos do df_real_test também
    # [Lógica omitida para brevidade, mas segue o mesmo padrão de treino/predição]
    
    print("🚀 Processo concluído. Verifique os arquivos .png na pasta.")