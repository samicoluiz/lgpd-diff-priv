import pandas as pd
import numpy as np

class TSEDataWrangler:
    def __init__(self):
        # Removi a duplicata de SG_PARTIDO e mantive a ordem lógica
        self.final_cols = [
            'SG_UF', 'NM_UE', 'CD_CARGO', 'NR_PARTIDO', 'SG_PARTIDO', 
            'CD_GENERO', 'CD_GRAU_INSTRUCAO', 'CD_ESTADO_CIVIL', 
            'CD_COR_RACA', 'CD_OCUPACAO', 'ANO_NASCIMENTO', 
            'DS_SITUACAO_CANDIDATURA', 'DS_SIT_TOT_TURNO'
        ]
        
        # Centralizei os limites aqui para facilitar o seu ajuste de TCC
        # Aumentamos NM_UE e SG_PARTIDO para forçar a dimensionalidade
        self.limits = {
            'NM_UE': 100,            # Top 100 cidades (Aumenta o risco/unicidade)
            'SG_PARTIDO': 35,       # Todos os partidos (Aumenta o risco/unicidade)
            'CD_OCUPACAO': 15,       # Mais profissões para o AIM aprender
            'NR_PARTIDO': 15,
            'SG_UF': 27,            # Todos os estados
            'ANO_NASCIMENTO': 4      # Mantemos 4 categorias de bining
        }
        self.default_limit = 6 

    def process(self, df):
        # 1. Binning de Idade (Transformação de dado contínuo em categórico)
        if 'DT_NASCIMENTO' in df.columns:
            years = pd.to_datetime(df['DT_NASCIMENTO'], errors='coerce').dt.year
            years = years.fillna(years.median())

            bins = [0, 1970, 1985, 2000, 2026]
            labels = ['VETERANO', 'EXPERIENTE', 'JOVEM_ADULTO', 'NOVA_GERACAO']
            df['ANO_NASCIMENTO'] = pd.cut(years, bins=bins, labels=labels).astype(str)
        else:
            df['ANO_NASCIMENTO'] = "NAO_INFORMADO"

        # 2. Seleção de Colunas
        available_cols = [c for c in self.final_cols if c in df.columns]
        df_reduced = df[available_cols].copy()

        # 3. Redução de Cardinalidade (Top-N + OUTROS)
        for col in df_reduced.columns:
            df_reduced[col] = df_reduced[col].astype(str).str.strip().str.upper()
            
            # AGORA USA OS LIMITES DO __INIT__
            limit = self.limits.get(col, self.default_limit)
            
            if df_reduced[col].nunique() > limit:
                top_items = df_reduced[col].value_counts().nlargest(limit).index
                df_reduced[col] = df_reduced[col].apply(
                    lambda x: x if x in top_items else "OUTROS"
                )
        
        return df_reduced

def apply_wrangling(df):
    wrangler = TSEDataWrangler()
    return wrangler.process(df)