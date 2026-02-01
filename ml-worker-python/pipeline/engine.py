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
from anonymeter.evaluators import SinglingOutEvaluator
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .wrangling_tse import apply_wrangling

class PrivacyEngine:
    def __init__(self):
        # 1. Configuração do motor NLP (Português e Inglês)
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
        
        # Inicializamos o Analyzer com o motor configurado
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
        
        # 2. Adição de Regras Customizadas (Ex: CPF)
        self._add_cpf_recognizer()
        
        # 3. Estado do Engine e Hardware
        self.last_df_clean = None  
        self.synth_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Motor configurado para PORTUGUÊS usando: {self.device}")

    # --- MÉTODO MAESTRO (ORQUESTRAÇÃO) ---

    def run_pipeline(self, input_path, epsilon=1.0):
        """Executa o pipeline completo de geração e auditoria."""
        try:
            # 1. Ingestão
            df_raw = self._load_data(input_path)
            df_working = self._sample_data(df_raw)
            
            # 2. Pré-processamento e Limpeza (Wrangling + PII)
            df_clean, pii_cols = self._preprocess_and_clean(df_working)
            self.last_df_clean = df_clean.copy() 

            # 3. Núcleo de Síntese (Treinamento e Geração)
            train_time = self._train_model(df_clean, epsilon)
            df_synthetic, gen_time = self._generate_data(df_clean)

            # 4. Auditoria de Resultados (Ablação e Utilidade)
            scores = self._audit_results(df_working, df_clean, df_synthetic, epsilon, train_time, gen_time)

            # 5. Exportação
            output_path = self._save_output(df_synthetic, input_path)
            
            return output_path, (1.0 - scores['privacy_synth']), pii_cols, scores['util_marginal']

        except Exception as e:
            print(f"[ERROR] Falha crítica no Pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", 1.0, [], 0.0

    # --- MÉTODOS PRIVADOS (TRABALHADORES) ---

    def _load_data(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.parquet':
            return pd.read_parquet(path)
        return pd.read_csv(path, sep=';', encoding='iso-8859-1', low_memory=False)

    def _sample_data(self, df):
        if len(df) > 100000:
            print(f"[INFO] Dataset grande. Amostrando 100.000 linhas.")
            return df.sample(n=100000, random_state=42).reset_index(drop=True)
        return df.copy()

    def _preprocess_and_clean(self, df):
        print("[WRANGLING] Aplicando limpeza e redução de cardinalidade...")
        df_wrangled = apply_wrangling(df)
        self.analyze_cardinality(df_wrangled)
        
        pii_cols = self.detect_pii_columns(df_wrangled)
        return df_wrangled.drop(columns=pii_cols), pii_cols

    def _train_model(self, df_clean, epsilon):
        self.synth_model = Plugins().get(
            "aim", 
            epsilon=float(epsilon), 
            delta=1e-6,
            max_cells=200000, 
            degree=2,
            device=self.device,
            random_state=42
        )
        print(f"[IA] Treinando AIM em {self.device}...")
        start = time.perf_counter()
        self.synth_model.fit(df_clean)
        
        # Persistência do modelo gerador para auditoria
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.synth_model, f"models/aim_model_eps_{epsilon}.joblib")
        
        return time.perf_counter() - start

    def _generate_data(self, df_clean):
        print(f"[IA] Gerando dados sintéticos...")
        start = time.perf_counter()
        df_gen = self.synth_model.generate(count=len(df_clean)).dataframe()
        return df_gen, time.perf_counter() - start

    def _audit_results(self, df_working, df_clean, df_synthetic, epsilon, t_train, t_gen):
        # 1. Estudo de Ablação (Wrangling vs AIM)
        score_wrangling, score_synth, dp_gain = self.evaluate_privacy_gain(df_working, df_clean, df_synthetic)
        
        # 2. Utilidade Estatística
        util_marginal, util_joint = self.calculate_utility(df_clean, df_synthetic)
        
        # 3. Log Consolidado
        self.log_execution(
            epsilon, score_synth, score_wrangling, dp_gain,
            util_marginal, util_joint, df_clean, t_train, t_gen
        )
        
        return {
            'privacy_synth': score_synth,
            'util_marginal': util_marginal
        }

    def _save_output(self, df_synth, input_path):
        output_path = input_path.replace("raw", "synthetic").replace(".csv", ".parquet")
        df_synth.to_parquet(output_path)
        return output_path

    # --- MÉTODOS DE APOIO TÉCNICO ---

    def _add_cpf_recognizer(self):
        cpf_pattern = Pattern(name="cpf_pattern", regex=r"\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11}", score=0.8)
        cpf_recognizer = PatternRecognizer(supported_entity="CPF", patterns=[cpf_pattern], supported_language="pt")
        self.analyzer.registry.add_recognizer(cpf_recognizer)

    def analyze_cardinality(self, df):
        print("\n--- 📊 ANÁLISE DE CARDINALIDADE (ORÇAMENTO DE PRIVACIDADE) ---")
        for col in df.columns:
            n_unique = df[col].nunique()
            status = "✅ OK" if n_unique < 20 else "⚠️ ALTA"
            if n_unique > 50: status = "🚨 CRÍTICA"
            print(f"Colun: {col:<20} | Categorias: {n_unique:<5} | Status: {status}")
        print("-----------------------------------------------------------\n")

    def calculate_utility(self, df_ori, df_syn):
        common_cols = [c for c in df_ori.columns if c in df_syn.columns]
        
        # Marginal (1-Way)
        marginal_jsds = []
        for col in common_cols:
            p = df_ori[col].value_counts(normalize=True).sort_index()
            q = df_syn[col].value_counts(normalize=True).sort_index()
            p, q = p.align(q, fill_value=0)
            marginal_jsds.append(jensenshannon(p, q, base=2))
        util_marginal = 1.0 - np.mean(marginal_jsds)

        # Joint (2-Way Pairwise)
        pair_jsds = []
        all_pairs = list(itertools.combinations(common_cols, 2))
        for col1, col2 in all_pairs[:15]: # Analisando os top 15 pares
            p = df_ori.groupby([col1, col2]).size() / len(df_ori)
            q = df_syn.groupby([col1, col2]).size() / len(df_syn)
            p, q = p.align(q, fill_value=0)
            pair_jsds.append(jensenshannon(p, q, base=2))
        
        util_joint = 1.0 - np.mean(pair_jsds)
        return util_marginal, util_joint

    def detect_pii_columns(self, df):
        pii_cols = []
        sample_size = min(50, len(df))
        sample_df = df.head(sample_size)
        
        print(f"[INFO] Inspecionando {sample_size} amostras para conformidade LGPD...")

        for col in df.columns:
            pii_hits = 0
            for val in sample_df[col]:
                text = str(val).strip()
                if not text or text.lower() == 'nan': continue
                results = self.analyzer.analyze(text=text, language='pt', entities=[])
                if len(results) > 0: pii_hits += 1

            if pii_hits > (sample_size * 0.1):
                print(f"[WARN] [ALERTA LGPD] PII detectada em '{col}'")
                pii_cols.append(col)
        return pii_cols

    def evaluate_privacy_gain(self, df_orig, df_clean, df_synth):
        print("[EVAL] Iniciando Ataque Simulado (Ablação)...")
        # Amostra comum para os dois ataques
        eval_orig = df_orig.sample(n=min(2000, len(df_orig)), random_state=42)
        
        # 1. Risco do dado apenas Limpo (Wrangling)
        ev_clean = SinglingOutEvaluator(ori=eval_orig, syn=df_clean)
        ev_clean.evaluate()
        score_clean = 1.0 - ev_clean.risk().value

        # 2. Risco do dado Sintético (AIM)
        ev_synth = SinglingOutEvaluator(ori=eval_orig, syn=df_synth)
        ev_synth.evaluate()
        score_synth = 1.0 - ev_synth.risk().value

        gain = score_synth - score_clean
        print(f"--- 🛡️  PRIVACIDADE: Limpo {score_clean:.4f} | DP {score_synth:.4f} | Ganho {gain:.4f}")
        
        return score_clean, score_synth, gain

    def log_execution(self, epsilon, privacy_score, score_clean, dp_gain, util_marginal, util_joint, df_after_wrangling, train_time, gen_time):
        log_file = "experiments_log.csv"
        file_exists = os.path.isfile(log_file)
        
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon": epsilon,
            "score_total": f"{privacy_score:.4f}",
            "score_wrangling": f"{score_clean:.4f}",
            "dp_gain": f"{dp_gain:.4f}",
            "util_marginal": f"{util_marginal:.4f}",
            "util_joint": f"{util_joint:.4f}",
            "train_sec": f"{train_time:.2f}",
            "gen_sec": f"{gen_time:.2f}",
            "rows": len(df_after_wrangling)
        }

        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)