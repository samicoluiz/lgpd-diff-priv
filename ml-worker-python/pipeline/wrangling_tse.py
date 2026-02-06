import pandas as pd
import numpy as np

class TSEDataWrangler:
    def __init__(self, strategy="intensive"):
        self.strategy = strategy
        
        # 1. BLACKLIST: Identificadores que impossibilitam a anonimização diferencial
        # Se esses campos entrarem no modelo, o risco de re-identificação é 100%
        self.blacklist = [
            'NR_CPF_CANDIDATO', 'NM_CANDIDATO', 'NM_SOCIAL_CANDIDATO', 
            'NR_TITULO_ELEITORAL_CANDIDATO', 'SQ_CANDIDATO', 'NM_EMAIL',
            'NR_PROCESSO', 'NR_CANDIDATO', 'NM_URNA_CANDIDATO'
        ]

        # 2. COLUNAS DE INTERESSE: O que realmente importa para a análise
        self.base_cols = [
            'SG_UF', 'NM_UE', 'CD_CARGO', 'NR_PARTIDO', 'SG_PARTIDO', 
            'CD_GENERO', 'CD_GRAU_INSTRUCAO', 'CD_ESTADO_CIVIL', 
            'CD_COR_RACA', 'CD_OCUPACAO', 'FAIXA_ETARIA', 
            'DS_SITUACAO_CANDIDATURA', 'DS_SIT_TOT_TURNO'
        ]

        # 3. LIMITES DE CARDINALIDADE (Top-N): 
        # Quanto menor o número, maior a privacidade (e menor o risco na GUI)
        if strategy == "high_fidelity":
            self.limits = {
                'NM_UE': 3000,          # Exposição proposital para testes
                'CD_OCUPACAO': 245,     
                'SG_PARTIDO': 45,       
                'ANO_NASCIMENTO': 80
            }
            self.default_limit = 100
        elif strategy == "minimal":
            self.limits = {
                'NM_UE': 500,
                'CD_OCUPACAO': 100,
                'FAIXA_ETARIA': 15
            }
            self.default_limit = 40
        else: # INTENSIVE (O modo "Cofre" para o TCC)
            self.limits = {
                'NM_UE': 50,            # Mantém apenas as 50 maiores cidades (capitais/polos)
                'CD_OCUPACAO': 10,      # Agrupa centenas de profissões em 10 grupos principais
                'SG_PARTIDO': 30,       
                'NR_PARTIDO': 30,
                'FAIXA_ETARIA': 6       # Apenas 6 grandes grupos geracionais
            }
            self.default_limit = 10 

    def process(self, df):
        # Cópia para evitar SettingWithCopyWarning
        df = df.copy()

        # --- A. REMOÇÃO DE PII DIRETA ---
        df = df.drop(columns=[c for c in self.blacklist if c in df.columns])

        # --- B. GENERALIZAÇÃO TEMPORAL (IDADE) ---
        if 'DT_NASCIMENTO' in df.columns:
            # Converte para data e extrai o ano
            years = pd.to_datetime(df['DT_NASCIMENTO'], errors='coerce').dt.year
            years = years.fillna(years.median())

            if self.strategy == "high_fidelity":
                # Mantém o ano exato (Alto risco de inferência)
                df['FAIXA_ETARIA'] = years.astype(int).astype(str)
            else:
                # Intensive: Binning geracional (Derruba o Inference Attack)
                bins = [0, 1960, 1975, 1985, 1995, 2005, 2026]
                labels = ['BOOMER', 'GEN_X', 'MILLENNIAL_FALDA', 'MILLENNIAL_NOVO', 'GEN_Z', 'NEW_GEN']
                df['FAIXA_ETARIA'] = pd.cut(years, bins=bins, labels=labels).astype(str)
        else:
            df['FAIXA_ETARIA'] = "NAO_INFORMADO"

        # --- C. SELEÇÃO E LIMPEZA DE COLUNAS ---
        available_cols = [c for c in self.base_cols if c in df.columns]
        df = df[available_cols].copy()

        # --- D. REDUÇÃO DE CARDINALIDADE (O "GROSSO" DA ANONIMIZAÇÃO) ---
        for col in df.columns:
            # Padronização para evitar duplicidade (ex: "MÉDICO" vs "medico")
            df[col] = df[col].astype(str).str.strip().str.upper().replace('NAN', 'NULL')
            
            # Pula redução se for modo fidelidade total para certas colunas
            if self.strategy == "high_fidelity" and col in ['NM_UE', 'CD_OCUPACAO']:
                continue

            limit = self.limits.get(col, self.default_limit)
            
            # Se a coluna tiver mais categorias que o permitido, agrupamos o "resto"
            if df[col].nunique() > limit:
                top_items = df[col].value_counts().nlargest(limit).index
                df[col] = df[col].apply(lambda x: x if x in top_items else "OUTROS_GRUPOS")

        return df

def apply_wrangling(df, strategy="intensive"):
    """
    Função de conveniência para o engine.py
    """
    wrangler = TSEDataWrangler(strategy=strategy)
    return wrangler.process(df)