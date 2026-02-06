import os, time, threading, sys, warnings, pandas as pd
from synthcity.plugins import Plugins

os.environ["LOGURU_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")

class Heartbeat:
    def __init__(self):
        self.active = False
    def _spin(self):
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i, start_t = 0, time.time()
        while self.active:
            elapsed = (time.time() - start_t) / 60
            sys.stdout.write(f"\r {chars[i % len(chars)]} AIM Otimizando Marginais... ({elapsed:.2f} min)")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    def start(self):
        self.active = True
        threading.Thread(target=self._spin).start()
    def stop(self):
        self.active = False
        sys.stdout.write("\r" + " " * 70 + "\r")

if __name__ == "__main__":
    print("\n🧬 SÍNTESE DO WORKLOAD ORIGINAL")
    df_train = pd.read_parquet("df_real_train.parquet")
    hb = Heartbeat()

    for eps in [10.0, 1.0, 0.1]:
        print(f"🚀 Iniciando Epsilon {eps}...")
        hb.start()
        start = time.time()
        
        syn_model = Plugins().get("aim", epsilon=eps)
        syn_model.fit(df_train)
        df_syn = syn_model.generate(count=len(df_train)).dataframe()
        
        df_syn.to_parquet(f"df_syn_eps_{eps}.parquet")
        hb.stop()
        print(f"✅ Epsilon {eps} pronto em {(time.time()-start)/60:.2f} min.")