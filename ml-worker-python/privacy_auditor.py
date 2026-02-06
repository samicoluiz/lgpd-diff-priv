import pandas as pd
import warnings
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator

warnings.filterwarnings("ignore")

class PrivacyAuditor:
    def __init__(self, df_real, df_syn, control_cols, target_col, sample_size=2500):
        # Amostragem para garantir que o teste termine em tempo hábil
        self.sample_size = min(sample_size, len(df_real))
        
        self.df_real = df_real.sample(self.sample_size, random_state=42).astype(str)
        self.df_syn = df_syn.sample(self.sample_size, random_state=42).astype(str)
        
        self.control_cols = control_cols
        self.target_col = target_col
        self.results = {}

    def run_all_attacks(self, n_attacks=300):
        print(f"🕵️ Auditoria Turbo: Amostra de {self.sample_size} registros.")
        print(f"🛠️ Configuração: {n_attacks} ataques por vetor.")

        # 1. SINGLING OUT
        print("   - Atacando Singling Out... ", end="", flush=True)
        eval_so = SinglingOutEvaluator(ori=self.df_real, syn=self.df_syn, n_attacks=n_attacks)
        eval_so.evaluate()
        self.results['Singling Out'] = eval_so.risk()
        print("✅")

        # 2. LINKABILITY
        print("   - Atacando Linkability... ", end="", flush=True)
        eval_link = LinkabilityEvaluator(ori=self.df_real, syn=self.df_syn, 
                                          aux_cols=self.control_cols, n_attacks=n_attacks)
        eval_link.evaluate()
        self.results['Linkability'] = eval_link.risk()
        print("✅")

        # 3. INFERENCE
        print("   - Atacando Inference... ", end="", flush=True)
        # AJUSTE DEFINITIVO: O parâmetro esperado é 'secret'
        eval_inf = InferenceEvaluator(ori=self.df_real, 
                                       syn=self.df_syn, 
                                       aux_cols=self.control_cols, 
                                       secret=self.target_col, # O 'alvo' agora é o 'secret'
                                       n_attacks=n_attacks)
        eval_inf.evaluate()
        self.results['Inference'] = eval_inf.risk()
        print("✅")

    def print_summary(self, epsilon):
        print(f"\n📊 RESULTADOS DE PRIVACIDADE (Epsilon {epsilon})")
        print("-" * 45)
        for attack, risk in self.results.items():
            # Exibindo valor do risco e intervalo de confiança
            print(f"🔹 {attack:15} | Risco: {risk.value:.4f} | IC: ({risk.ci[0]:.4f}, {risk.ci[1]:.4f})")
        print("-" * 45)

# ... (mantenha os imports e a classe PrivacyAuditor igual) ...

if __name__ == "__main__":
    QUASI_IDS = ['CD_CARGO', 'SG_PARTIDO', 'CD_GENERO']
    TARGET = 'DS_SIT_TOT_TURNO'
    
    # Configurações de Rigor Acadêmico Equilibrado
    SAMPLE_SIZE = 2000  # Diminuímos para 2.5k para fluir
    N_ATTACKS = 150     # 300 ataques já dão um IC (Intervalo de Confiança) sólido
    
    try:
        df_real = pd.read_parquet("df_real_auditoria.parquet")
        
        for eps in [0.1, 1.0, 10.0]:
            filename = f"df_syn_eps_{eps}.parquet"
            try:
                df_syn = pd.read_parquet(filename)
                print(f"\n🚀 Analisando Epsilon {eps}...")
                
                # Criamos o auditor com a amostra equilibrada
                auditor = PrivacyAuditor(df_real, df_syn, 
                                         control_cols=QUASI_IDS, 
                                         target_col=TARGET, 
                                         sample_size=SAMPLE_SIZE)
                
                # Rodamos com 300 ataques (Equilíbrio entre tempo e precisão)
                auditor.run_all_attacks(n_attacks=N_ATTACKS) 
                auditor.print_summary(epsilon=eps)
                
            except FileNotFoundError:
                print(f"⚠️ Arquivo {filename} não encontrado.")
                
    except FileNotFoundError:
        print("❌ Erro: df_real_auditoria.parquet não encontrado.")