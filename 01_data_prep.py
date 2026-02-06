import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    PATH = "backend-go/data/raw_consulta_cand_2024_BRASIL.parquet"
    
    # Mantendo o máximo de colunas úteis (sem IDs/Nomes)
    cols = [
        'DS_GENERO', 'DS_GRAU_INSTRUCAO', 'DS_ESTADO_CIVIL', 'DS_COR_RACA',
        'SG_PARTIDO', 'DS_OCUPACAO', 'SG_UF', 'DS_SIT_TOT_TURNO'
    ]
    
    df = pd.read_parquet(PATH, columns=cols).dropna()
    
    # Criando o Alvo Real
    df['ALVO'] = df['DS_SIT_TOT_TURNO'].apply(lambda x: 1 if 'ELEITO' in str(x).upper() else 0)
    df = df.drop(columns=['DS_SIT_TOT_TURNO'])

    # O df_test é o "Gabarito Real". Ele NUNCA entra no AIM.
    train, test = train_test_split(df.sample(20000), test_size=0.2, random_state=42)
    
    train.to_parquet("df_real_train.parquet")
    test.to_parquet("df_real_test.parquet")
    
    print(f"✅ Bases prontas. Treino: {len(train)} | Teste Real (Gabarito): {len(test)}")