import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define o backend não-interativo antes de importar pyplot
import matplotlib
matplotlib.use('Agg') 

def generate_tcc_plots(log_path="experiments_log.csv", output_path="resultado_epsilon_impact.png"):
    if not os.path.exists(log_path):
        print(f"[VISUALIZER] Arquivo {log_path} não encontrado para gerar gráficos.")
        return

    try:
        df = pd.read_csv(log_path)
        
        # Garante que os tipos estão corretos para o gráfico
        df['epsilon'] = pd.to_numeric(df['epsilon'])
        df['utility_score'] = pd.to_numeric(df['utility_score'])
        df['privacy_score'] = pd.to_numeric(df['privacy_score'])

        # Ordenação por epsilon para manter a lógica da linha
        df = df.sort_values('epsilon')

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plotagem das métricas
        plt.plot(df['epsilon'], df['utility_score'], marker='o', label='Utilidade (JSD)', color='#10b981', linewidth=2)
        plt.plot(df['epsilon'], df['privacy_score'], marker='s', label='Privacidade (Risk)', color='#6366f1', linewidth=2)

        # Configurações de escala e labels
        plt.xscale('log') 
        plt.title('Trade-off: Privacidade vs Utilidade (Modelo AIM)', fontsize=14)
        plt.xlabel('Orçamento de Privacidade (Epsilon) - Escala Log', fontsize=12)
        plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() # Libera a memória da figura
        print(f"[VISUALIZER] Gráfico atualizado: {output_path}")
        
    except Exception as e:
        print(f"[VISUALIZER] Erro ao gerar gráfico: {e}")