# AI Privacy & Synthetic Data Engine

Este projeto implementa uma arquitetura distribuída de alta performance para a geração de dados sintéticos com **Privacidade Diferencial (DP)** e detecção de **PII (Informações Pessoais Identificáveis)** sob a ótica da **LGPD**. 

A solução combina a eficiência do **Go** para orquestração com o ecossistema de Machine Learning do **Python**, utilizando **gRPC** para comunicação de baixa latência, **NVIDIA CUDA** para processamento pesado e **DuckDB** para análise colunar ultrarrápida.

---

## Tech Stack & Diferenciais

| Camada | Tecnologia | Papel Principal |
| :--- | :--- | :--- |
| **Orquestração** | **Go (Golang)** | Servidor de borda, concorrência e gerenciamento de tarefas gRPC. |
| **Frontend** | **HTMX + Templ** | Interface reativa com SSR para visualização de métricas de privacidade. |
| **Motor de IA** | **Python (PyTorch)** | Geração via **AIM** com aceleração por hardware (**NVIDIA RTX 4080 Super**). |
| **Processamento NLP** | **Presidio + SpaCy** | Reconhecimento de entidades brasileiras com modelo `pt_core_news_lg`. |
| **Privacidade** | **Anonymeter** | Auditoria de segurança contra ataques de *Singling-out* e *Linkability*. |
| **Data Layer** | **DuckDB & Parquet** | Armazenamento colunar eficiente e consultas SQL de alta performance. |

---

## Arquitetura e Engenharia de Dados

O sistema opera através de um pipeline otimizado para grandes volumes e conformidade legal:



1.  **Ingestão & Scan LGPD:** O backend recebe datasets e o worker Python realiza o reconhecimento de PII utilizando um motor NLP configurado para **Português Brasileiro**, incluindo reconhecedores customizados para **CPF**.
2.  **Discretização por Quantis:** Para viabilizar a **Privacidade Diferencial ($\epsilon$)** em colunas de alta cardinalidade, o sistema aplica discretização automática. Isso reduz o tamanho do domínio (*domain size*), prevenindo a memorização de valores exatos e garantindo a convergência do modelo.
3.  **Síntese Acelerada (CUDA):** O modelo **AIM (Adaptive Independent Mechanisms)** é treinado na GPU para capturar as correlações estatísticas do dataset original, gerando uma versão sintética que preserva a utilidade para fins de BI e Machine Learning.
4.  **Auditoria de Risco:** O sistema gera um **Score de Privacidade** baseado na probabilidade de sucesso de ataques de re-identificação, validando a eficácia da proteção via **Anonymeter**.

---

## Considerações de Implementação

### Estratégia de Discretização
Para evitar erros de explosão de memória em domínios numéricos contínuos, os dados são agrupados em *bins* (intervalos).
* **Impacto no ML:** Os intervalos atuam como uma forma de regularização, tornando os classificadores treinados nestes dados mais robustos contra ruído, embora exijam codificação ordinal ou *midpoint* para processamento numérico posterior.



### Aceleração por Hardware
O projeto está configurado para utilizar núcleos **CUDA** em ambientes com GPUs NVIDIA. O treinamento de redes neurais e os mecanismos de DP são movidos da CPU para o hardware dedicado, reduzindo drasticamente o tempo de síntese em datasets massivos.

---

## Configuração do Ambiente

* **WSL2 (Ubuntu):** Otimizado para integração com Docker e NVIDIA Container Toolkit.
* **Modelos NLP:** Requer o download do modelo `pt_core_news_lg` (`python -m spacy download pt_core_news_lg`) para processamento nativo em português.
* **Storage:** Utiliza volumes compartilhados para troca de arquivos Parquet, evitando o overhead de transferência de dados pesados via rede.

---

## Comandos Principais

* `make setup-venv`: Configura o ambiente Python e baixa modelos de linguagem.
* `make gen-proto`: Compila as definições do gRPC para Go e Python.
* `python test_client.py`: Executa um teste de fumaça simulando uma requisição de anonimização.
* `docker-compose up --build`: Levanta os serviços com suporte a GPU e volumes de dados.