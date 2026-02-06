from anonymeter.evaluators import InferenceEvaluator

def run_system_audit(df_ori, df_obs, aux_cols):
    targets = ['CD_COR_RACA', 'CD_GRAU_INSTRUCAO', 'CD_ESTADO_CIVIL']
    results = {}

    print("--- 🛡️  AUDITORIA DE SISTEMA (INFERÊNCIA MULTI-ALVO) ---")
    
    for target in targets:
        eval_inf = InferenceEvaluator(ori=df_ori, syn=df_obs, aux_cols=aux_cols, secret=target)
        eval_inf.evaluate()
        results[target] = eval_inf.risk().value
        print(f"Target: {target} | Risco: {results[target]:.4f}")

    avg_risk = sum(results.values()) / len(results)
    max_risk = max(results.values())
    
    print("-" * 50)
    print(f"Risco Médio do Sistema: {avg_risk:.4f}")
    print(f"Risco Crítico (Worst-case): {max_risk:.4f}")
    return avg_risk, max_risk