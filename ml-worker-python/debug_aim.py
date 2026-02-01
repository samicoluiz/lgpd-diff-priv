import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
import sys
import warnings

warnings.filterwarnings("ignore")

def test_aim_epsilon():
    print("--- 🧪 Teste Isolado do AIM (Synthcity) ---")
    
    # Criar um dataset dummy simples
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, 100),
        'salary': np.random.randint(2000, 20000, 100),
        'group': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Dataset de teste criado (100 linhas).")

    # Teste 1: Epsilon = 0.1 (Muita privacidade, muito ruído)
    print("\n👉 Treinando com Epsilon = 0.1...")
    model_low = Plugins().get("aim", epsilon=0.1, delta=1e-5)
    model_low.fit(df)
    syn_low = model_low.generate(count=100).dataframe()
    print("Gerado (Low Epsilon).")

    # Teste 2: Epsilon = 100.0 (Baixa privacidade, pouco ruído - Quase fiel)
    print("\n👉 Treinando com Epsilon = 100.0...")
    model_high = Plugins().get("aim", epsilon=100.0, delta=1e-5)
    model_high.fit(df)
    syn_high = model_high.generate(count=100).dataframe()
    print("Gerado (High Epsilon).")

    # Comparação simples (Médias)
    mean_real = df['age'].mean()
    mean_low = syn_low['age'].mean()
    mean_high = syn_high['age'].mean()

    print(f"\n📊 Resultados (Média da coluna 'age'):")
    print(f"Original: {mean_real:.2f}")
    print(f"Epsilon 0.1: {mean_low:.2f} (Diferença: {abs(mean_real - mean_low):.2f})")
    print(f"Epsilon 100: {mean_high:.2f} (Diferença: {abs(mean_real - mean_high):.2f})")

    if abs(mean_real - mean_high) < abs(mean_real - mean_low):
        print("\n✅ CONCLUSÃO: O parâmetro Epsilon PARECE estar funcionando (O erro diminuiu com epsilon maior).")
    else:
        print("\n❌ CONCLUSÃO: O parâmetro Epsilon NÃO PARECE afetar o resultado como esperado.")

    # Verificar se os dataframes são idênticos (bit a bit)
    if syn_low.equals(syn_high):
         print("❌ CRÍTICO: Os dados gerados são IDÊNTICOS para epsilons diferentes!")
    else:
         print("✅ Os dados gerados são diferentes.")

if __name__ == "__main__":
    test_aim_epsilon()
