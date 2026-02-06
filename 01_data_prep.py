import pandas as pd
import time, sys, threading, os
from sklearn.model_selection import train_test_split

class Heartbeat:
    def __init__(self, message="Processando"):
        self.message, self.active = message, False
    def _spin(self):
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while self.active:
            sys.stdout.write(f"\r {chars[i % len(chars)]} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    def start(self): self.active = True; threading.Thread(target=self._spin).start()
    def stop(self): self.active = False; sys.stdout.write("\r" + " " * 80 + "\r")

if __name__ == "__main__":
    hb = Heartbeat("🧪 FORÇANDO SENSIBILIDADE: Reduzindo para 5k registros")
    hb.start()
    
    PATH = "backend-go/data/raw_consulta_cand_2024_BRASIL.parquet"
    
    # Workload de Alta Dimensão
    cols = [
    'DS_GENERO', 'DS_GRAU_INSTRUCAO', 'DS_ESTADO_CIVIL', 'DS_COR_RACA', 
    'SG_PARTIDO', 'TP_AGREMIACAO', 'DS_CARGO', 'DS_SITUACAO_CANDIDATURA', 
    'DS_OCUPACAO', 'SG_UF', 'DS_SIT_TOT_TURNO' # <--- Adicionamos OCUPACAO
    ]
    
    df = pd.read_parquet(PATH, columns=cols).dropna()
    
    # Alvo
    df['ALVO'] = df['DS_SIT_TOT_TURNO'].apply(lambda x: 1 if 'ELEITO' in str(x).upper() else 0)
    df = df.drop(columns=['DS_SIT_TOT_TURNO'])
    
    # O SEGREDO DO TRADE-OFF: Menos dados = Ruído Laplace mais destrutivo
    df_sample = df.sample(n=5000, random_state=42)
    
    train, test = train_test_split(df_sample, test_size=0.2, random_state=42, stratify=df_sample['ALVO'])
    
    train.to_parquet("df_real_train.parquet")
    test.to_parquet("df_real_test.parquet")
    
    hb.stop()
    print(f"✅ Stress-Test Pronto! Treino: {len(train)} registros.")