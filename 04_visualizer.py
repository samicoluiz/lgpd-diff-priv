import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_final_plot():
    # Dados extraídos do seu log final
    epsilons = [50.0, 35.0, 20.0, 10.0, 1.0, 0.1, 0.001]
    f1_puro = [0.5011, 0.4942, 0.5063, 0.4910, 0.4900, 0.4940, 0.5084]
    f1_wrang = [0.5010, 0.4942, 0.5010, 0.5111, 0.4893, 0.5118, 0.4910]
    priv_risk = [0.1613, 0.1563, 0.1284, 0.1394, 0.0995, 0.1135, 0.1055]
    
    base_real_puro = 0.5966
    base_real_wrang = 0.5759

    x = np.arange(len(epsilons))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- EIXO 1: UTILIDADE ---
    ax1.set_xlabel('Nível de Privacidade (Epsilon $\epsilon$)', fontweight='bold')
    ax1.set_ylabel('Utilidade (F1-Score)', color='tab:blue', fontweight='bold')
    
    # Linhas de Baseline Real
    ax1.axhline(y=base_real_puro, color='green', linestyle='--', alpha=0.6, label='Real Puro (Teto)')
    ax1.axhline(y=base_real_wrang, color='darkgreen', linestyle=':', alpha=0.6, label='Real Wrangled')
    
    # Linhas de Utilidade Sintética
    ax1.plot(x, f1_puro, marker='o', ls='-', color='skyblue', linewidth=2, label='Syn Puro')
    ax1.plot(x, f1_wrang, marker='s', ls='--', color='tab:blue', linewidth=2, label='Syn Wrangled')
    
    ax1.set_ylim(0.40, 0.65)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # --- EIXO 2: PRIVACIDADE ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Risco de Inferência (Anonymeter)', color='tab:red', fontweight='bold')
    ax2.plot(x, priv_risk, marker='D', ls='-', color='tab:red', linewidth=3, markersize=8, label='Risco de Privacidade')
    
    ax2.set_ylim(0, 0.30)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # --- AJUSTES FINAIS ---
    plt.title('Metodologia Samico: Avaliação de Utilidade vs. Privacidade\n(TSE 2024 - Algoritmo AIM)', fontsize=14, pad=20)
    plt.xticks(x, [str(e) for e in epsilons])
    
    # Unificar Legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    plt.grid(alpha=0.2)
    fig.tight_layout()
    plt.savefig('tradeoff_metodologia_samico.png', dpi=300, bbox_inches='tight')
    print("🚀 Gráfico 'tradeoff_metodologia_samico.png' gerado com sucesso!")

if __name__ == "__main__":
    generate_final_plot()