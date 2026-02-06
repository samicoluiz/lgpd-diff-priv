import pandas as pd
import torch
import os
import csv
import datetime
import time
import numpy as np
import joblib
import itertools
from scipy.spatial.distance import jensenshannon
from synthcity.plugins import Plugins
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Importação do wrangler ajustado
from .wrangling_tse import apply_wrangling

class PrivacyEngine:
    def __init__(self):
        # 1. Configuração do motor NLP para detecção de PII
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "pt", "model_name": "pt_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_lg"}
            ],
            "ner_model_configuration": {
                "labels_to_ignore": ["MISC", "ORG", "PER", "LOC"] 
            }
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
        
        self._add_cpf_recognizer()
        
        self.last_df_clean = None  
        self.synth_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Motor configurado para PORTUGUÊS usando: {self.device}")

    # --- MÉTODO MAESTRO ---

    def run_pipeline(self, input_path, epsilon=1.0):
        try:
            # 1. Carga e Amostragem (Garante performance no treinamento)
            df_raw = self._load_data(input_path)
            df_working = self._sample_data(df_raw)
            
            # 2. Preprocessamento Agressivo (Wrangling) e Detecção de PII
            # Alterado de "raw" para "intensive" para derrubar o risco de inferência na GUI
            df_clean, pii_cols = self._preprocess_and_clean(df_working, strategy="high_fidelity")
            self.last_df_clean = df_clean.copy() 

            # 3. Treinamento do Modelo Generativo (AIM - Adaptive Independence Model)
            train_time = self._train_model(df_clean, epsilon)
            
            # 4. Geração do Dataset Sintético
            df_synthetic, gen_time = self._generate_data(df_clean)

            # 5. Cálculo de Utilidade Estatística (Jensen-Shannon Distance)
            util_marginal, _ = self.calculate_utility(df_clean, df_synthetic)

            # 6. Salvamento do Resultado
            output_path = self._save_output(df_synthetic, input_path)
            
            print(f"[DONE] Pipeline de Geração Finalizado!")
            
            return output_path, df_clean, df_synthetic, pii_cols, util_marginal

        except Exception as e:
            print(f"[ERROR] Falha crítica no Pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", None, None, [], 0.0

    # --- MÉTODOS AUXILIARES ---

    def _load_data(self, path):
        """Carrega os dados tratando o encoding Latin-1 comum no TSE."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.parquet':
            return pd.read_parquet(path)
        # O padrão do TSE é ponto e vírgula com encoding ISO-8859-1
        return pd.read_csv(path, sep=';', encoding='iso-8859-1', low_memory=False)

    def _sample_data(self, df):
        """Limita o processamento a 100k linhas para viabilizar o treinamento em tempo real."""
        if len(df) > 100000:
            print(f"[INFO] Dataset grande ({len(df)} linhas). Amostrando 100.000 para o AIM.")
            return df.sample(n=10000, random_state=42).reset_index(drop=True)
        return df.copy()

    def _preprocess_and_clean(self, df, strategy="intensive"):
        """Aplica as regras de generalização e detecta colunas sensíveis."""
        print(f"[WRANGLING] Aplicando estratégia: {strategy.upper()}")
        
        # Chama o wrangler que criamos para o TSE
        df_wrangled = apply_wrangling(df, strategy=strategy)
        
        # Analisa cardinalidade para o log do terminal
        self.analyze_cardinality(df_wrangled)
        
        # Detecta PIIs remanescentes (como nomes que escaparam da lista)
        pii_cols = self.detect_pii_columns(df_wrangled)
        
        df_final = df_wrangled.drop(columns=pii_cols)
        print(f"[INFO] Colunas PII removidas: {pii_cols}")
        
        return df_final, pii_cols

    def _train_model(self, df_clean, epsilon):
        """Instancia e treina o plugin AIM do Synthcity."""
        self.synth_model = Plugins().get(
            "aim", 
            epsilon=float(epsilon), 
            delta=1e-6,
            max_cells=50000, 
            degree=2,
            device=self.device,
            random_state=42
        )
        print(f"[IA] Treinando AIM (Epsilon={epsilon}) em {self.device}...")
        start = time.perf_counter()
        self.synth_model.fit(df_clean)
        return time.perf_counter() - start

    def _generate_data(self, df_clean):
        """Gera os dados sintéticos respeitando o orçamento de privacidade."""
        print(f"[IA] Gerando dados sintéticos...")
        start = time.perf_counter()
        # count=len(df_clean) garante que o dataset sintético tenha o mesmo tamanho do original
        df_gen = self.synth_model.generate(count=len(df_clean)).dataframe()
        return df_gen, time.perf_counter() - start

    def calculate_utility(self, df_ori, df_syn):
        """Calcula a fidelidade estatística entre as bases."""
        common_cols = [c for c in df_ori.columns if c in df_syn.columns]
        marginal_jsds = []
        for col in common_cols:
            p = df_ori[col].value_counts(normalize=True).sort_index()
            q = df_syn[col].value_counts(normalize=True).sort_index()
            p, q = p.align(q, fill_value=0)
            marginal_jsds.append(jensenshannon(p, q, base=2))
        
        # Score de Utilidade: 1 - média das distâncias (Quanto mais perto de 1, melhor)
        util_marginal = 1.0 - np.mean(marginal_jsds)
        return util_marginal, 0.0

    def analyze_cardinality(self, df):
        """Log visual para identificar colunas que aumentam o risco de re-identificação."""
        print("\n--- 📊 ANÁLISE DE CARDINALIDADE (PÓS-WRANGLING) ---")
        for col in df.columns:
            n_unique = df[col].nunique()
            # Se uma coluna tem muitas categorias, o risco de inferência sobe
            status = "✅ SEGURO" if n_unique < 50 else "🚨 ALTO RISCO"
            print(f"Coluna: {col:<20} | Categorias: {n_unique:<5} | Status: {status}")
        print("-" * 50)

    def detect_pii_columns(self, df):
        """Usa o Microsoft Presidio para escanear amostras em busca de dados sensíveis."""
        pii_cols = []
        sample_df = df.head(min(100, len(df)))
        for col in df.columns:
            pii_hits = 0
            for val in sample_df[col]:
                results = self.analyzer.analyze(text=str(val), language='pt', entities=[])
                if len(results) > 0: pii_hits += 1
            # Se mais de 10% da amostra parecer PII, marca a coluna
            if pii_hits > (len(sample_df) * 0.1): 
                pii_cols.append(col)
        return pii_cols

    def _add_cpf_recognizer(self):
        """Adiciona suporte a CPFs ao analisador de PII."""
        cpf_pattern = Pattern(name="cpf_pattern", regex=r"\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11}", score=0.8)
        cpf_recognizer = PatternRecognizer(supported_entity="CPF", patterns=[cpf_pattern], supported_language="pt")
        self.analyzer.registry.add_recognizer(cpf_recognizer)

    def _save_output(self, df_synth, input_path):
        """Salva o dataset resultante em formato Parquet para preservar tipos de dados."""
        os.makedirs("output", exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_synthetic.parquet")
        output_path = os.path.join("output", filename)
        df_synth.to_parquet(output_path)
        return output_path