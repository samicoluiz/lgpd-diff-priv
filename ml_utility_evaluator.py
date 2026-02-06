import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class MLUtilityEvaluator:
    def __init__(self, target_col='CD_SITUACAO_CANDIDATO_TOT'):
        self.target_col = target_col
        self.le_map = {}

    def _preprocess(self, df):
        """Limpeza básica e encoding para os modelos de ML."""
        df = df.copy()
        # Remove colunas que não são features (IDs, nomes)
        cols_to_drop = ['SQ_CANDIDATO', 'NM_CANDIDATO', 'NR_CPF_CANDIDATO']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Encoding de colunas categóricas
        for col in df.select_dtypes(include=['object']).columns:
            if col not in self.le_map:
                self.le_map[col] = LabelEncoder()
                df[col] = self.le_map[col].fit_transform(df[col].astype(str))
            else:
                # Trata categorias novas no sintético que não existiam no real
                df[col] = df[col].map(lambda s: s if s in self.le_map[col].classes_ else self.le_map[col].classes_[0])
                df[col] = self.le_map[col].transform(df[col].astype(str))
        return df.fillna(0)

    def run_evaluation(self, df_real, df_syn):
        """
        Treina no sintético, testa no real. 
        Compara com o baseline (Treina no real, testa no real).
        """
        # Preparação das bases
        real_prep = self._preprocess(df_real)
        syn_prep = self._preprocess(df_syn)

        # Split do Real para teste (O teste é SEMPRE o dado real não visto)
        X_real = real_prep.drop(columns=[self.target_col])
        y_real = real_prep[self.target_col]
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42
        )

        # X e y do Sintético (usado integralmente para treino)
        X_train_syn = syn_prep.drop(columns=[self.target_col])
        y_train_syn = syn_prep[self.target_col]

        results = {}

        # Modelos para testar
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        for name, model in models.items():
            # 1. Baseline: Train Real -> Test Real
            model.fit(X_train_real, y_train_real)
            y_pred_base = model.predict(X_test_real)
            
            # 2. Experimento: Train Synthetic -> Test Real
            model.fit(X_train_syn, y_train_syn)
            y_pred_syn = model.predict(X_test_real)

            results[f"{name}_F1_Real"] = f1_score(y_test_real, y_pred_base, average='weighted')
            results[f"{name}_F1_Syn"] = f1_score(y_test_real, y_pred_syn, average='weighted')
            results[f"{name}_Acc_Delta"] = accuracy_score(y_test_real, y_pred_base) - accuracy_score(y_test_real, y_pred_syn)

        return results