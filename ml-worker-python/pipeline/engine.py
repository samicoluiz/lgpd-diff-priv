import pandas as pd
import torch
import os
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from synthcity.plugins import Plugins
from anonymeter.evaluators import SinglingOutEvaluator
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import SpacyNlpEngine

class PrivacyEngine:
    def __init__(self):
        # 1. Configuração do motor NLP para suportar PT e EN
        # Isso evita o erro de "No matching recognizers"
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "pt", "model_name": "pt_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_lg"}
            ],
        }
        
        # Criamos o motor NLP com a configuração acima
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        # Inicializamos o Analyzer com o motor configurado
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
        
        # 2. Adicionamos o reconhecedor de CPF (Regra Customizada)
        self._add_cpf_recognizer()
        
        # 3. Inicializamos o modelo AIM (Privacidade Diferencial)
        from synthcity.plugins import Plugins
        self.synth_model = Plugins().get(
            "aim", 
            epsilon=1.0, 
            delta=1e-5,
            max_cells=20000,
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Motor configurado para PORTUGUÊS usando: {self.device}")

    def _add_cpf_recognizer(self):
        cpf_pattern = Pattern(name="cpf_pattern", regex=r"\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11}", score=0.8)
        cpf_recognizer = PatternRecognizer(supported_entity="CPF", patterns=[cpf_pattern], supported_language="pt")
        self.analyzer.registry.add_recognizer(cpf_recognizer)

    def detect_pii_columns(self, df):
            pii_cols = []
            # Aumentamos a amostra para inspeção
            sample_size = min(50, len(df))
            sample_df = df.head(sample_size)
            
            print(f"[INFO] Inspecionando {sample_size} amostras por coluna para garantir conformidade LGPD...")

            for col in df.columns:
                pii_hits = 0
                detected_types = []

                for val in sample_df[col]:
                    text = str(val).strip()
                    if not text or text.lower() == 'nan':
                        continue
                    
                    # Analisamos cada célula da amostra
                    results = self.analyzer.analyze(text=text, language='pt', entities=[])
                    
                    if len(results) > 0:
                        pii_hits += 1
                        detected_types.append(results[0].entity_type)

                # Lógica de decisão: Se > 10% da amostra for PII, a coluna é sensível
                if pii_hits > (sample_size * 0.1):
                    most_common_type = max(set(detected_types), key=detected_types.count)
                    print(f"[WARN]  [ALERTA LGPD] PII confirmada em '{col}': {most_common_type} ({pii_hits} ocorrências)")
                    pii_cols.append(col)
                    
            return pii_cols
    
    def run_pipeline(self, input_path):
            try:
                # 1. Carregamento com tratamento de erro
                ext = os.path.splitext(input_path)[1].lower()
                print(f"[FILE] Processando: {os.path.basename(input_path)}")
                
                if ext == '.parquet':
                    df = pd.read_parquet(input_path)
                else:
                    df = pd.read_csv(input_path, sep=';', encoding='iso-8859-1')

                # [IMPORTANT] AMOSTRAGEM PARA O TCC (Segurança contra congelamento)
                # 30k linhas é o "sweet spot" para mostrar performance sem travar o gRPC
                if len(df) > 30000:
                    print(f"[WARN] Dataset muito grande ({len(df)} linhas). Amostrando 30.000 para viabilidade.")
                    df = df.sample(n=30000, random_state=42).reset_index(drop=True)

                # 2. Limpeza e Discretização
                # (Mantemos seu código de remoção de ID e discretização numérica)
                if 'id' in df.columns:
                    df = df.drop(columns=['id'])

                for col in df.select_dtypes(include=['number']).columns:
                    if df[col].nunique() > 20:
                        df[col] = pd.qcut(df[col], q=10, duplicates='drop').astype(str)
                
                pii_cols = self.detect_pii_columns(df)
                df_clean = df.drop(columns=pii_cols)

                # 3. Treinamento AIM (Agora sim na GPU se o KeOps estiver ok)
                print(f"[INFO] Treinando modelo AIM...")
                self.synth_model.fit(df_clean)
                
                print(f"[INFO] Gerando {len(df_clean)} registros sintéticos...")
                df_synthetic = self.synth_model.generate(count=len(df_clean)).dataframe()
                
                # 4. Avaliação de Risco Amostrada (Evita o travamento do Anonymeter)
                print("[INFO] Avaliando privacidade (Amostra de 1000 registros)...")
                eval_ori = df_clean.sample(n=min(1000, len(df_clean)))
                eval_syn = df_synthetic.sample(n=min(1000, len(df_synthetic)))
                
                evaluator = SinglingOutEvaluator(ori=eval_ori, syn=eval_syn)
                evaluator.evaluate()
                risk = evaluator.risk().value
                
                # 5. Salvamento e Retorno
                output_path = input_path.replace("raw", "synthetic").replace(".csv", ".parquet")
                df_synthetic.to_parquet(output_path)
                
                print(f"[SUCCESS] Processo concluído com sucesso!")
                return output_path, risk, pii_cols

            except Exception as e:
                print(f"[ERROR] ERRO NO PIPELINE: {str(e)}")
                # Retorna valores padrão para não quebrar o gRPC
                return "", 1.0, []