import os, time, threading, sys, warnings, pandas as pd
from synthcity.plugins import Plugins

os.environ["LOGURU_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")

class Heartbeat:
    def __init__(self): self.active = False
    def _spin(self):
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i, start_t = 0, time.time()
        while self.active:
            elapsed = (time.time() - start_t) / 60
            sys.stdout.write(f"\r {chars[i % len(chars)]} AIM Sintetizando... ({elapsed:.2f} min)")
            sys.stdout.flush()
            time.sleep(0.1); i += 1
    def start(self): self.active = True; threading.Thread(target=self._spin).start()
    def stop(self): self.active = False; sys.stdout.write("\r" + " " * 80 + "\r")

if __name__ == "__main__":
    df_train = pd.read_parquet("df_real_train.parquet")
    epsilons = [50.0, 35.0, 20.0, 10.0, 1.0, 0.1, 0.001]
    hb = Heartbeat()

    print(f"\n🧬 [SÍNTESE] Iniciando geração para {len(epsilons)} níveis...")

    for eps in epsilons:
        print(f"🚀 Processando Epsilon {eps}...")
        hb.start()
        syn_model = Plugins().get("aim", epsilon=eps)
        syn_model.fit(df_train)
        df_syn = syn_model.generate(count=len(df_train)).dataframe()
        df_syn.to_parquet(f"df_syn_eps_{eps}.parquet")
        hb.stop()