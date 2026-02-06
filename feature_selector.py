import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

class DPFeatureSelector:
    def __init__(self, target_col='DS_SIT_TOT_TURNO'):
        self.target_col = target_col

    def calculate_efficiency(self, df):
        print(f"📊 Calculando Eficiência DP para as colunas...")
        
        # Preparação rápida (Encoding para o algoritmo de MI)
        df_encoded = df.copy().astype(str)
        le = LabelEncoder()
        for col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        X = df_encoded.drop(columns=[self.target_col])
        y = df_encoded[self.target_col]
        
        # 1. Calcula a Informação Mútua Bruta (Sinal)
        mi_scores = mutual_info_classif(X, y, discrete_features=True, random_index=42)
        
        results = []
        for i, col in enumerate(X.columns):
            cardinality = df[col].nunique()
            mi = mi_scores[i]
            
            # 2. O SCORE DE EFICIÊNCIA DP:
            # Penaliza a cardinalidade de forma logarítmica (bits de informação)
            # Isso evita que colunas com 5000 municípios dominem o modelo
            efficiency = mi / np.log2(cardinality) if cardinality > 1 else 0
            
            results.append({
                'Feature': col,
                'MI_Raw': round(mi, 4),
                'Cardinality': cardinality,
                'DP_Efficiency': round(efficiency, 4)
            })
            
        return pd.DataFrame(results).sort_values(by='DP_Efficiency', ascending=False)