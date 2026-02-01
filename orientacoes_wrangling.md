Este documento estabelece as diretrizes técnicas para o pré-processamento de dados destinados ao algoritmo **AIM (Adaptive Iterative Method)**. O objetivo é otimizar o uso do orçamento de privacidade () e maximizar a utilidade estatística dos dados sintéticos gerados.

---

# Guia de Preparação de Dados para Modelagem AIM

## 1. O Princípio da Cardinalidade

O AIM opera selecionando e medindo "marginais" (correlações entre subconjuntos de colunas). A complexidade computacional e o consumo do orçamento de privacidade são diretamente proporcionais ao número de células combinadas:

Onde  é o número de categorias únicas na coluna .

### Ação Necessária:

* **Identificação de Colunas Críticas:** Colunas com cardinalidade superior a 20 devem ser avaliadas.
* **Agrupamento de Cauda Longa (Binning):** Categorias que representam uma fração insignificante do dataset (ex: menos de 1%) devem ser consolidadas em um rótulo genérico como "OUTROS". Isso evita que o ruído da Privacidade Diferencial destrua o sinal de categorias raras.

---

## 2. Redução de Dimensionalidade (Feature Selection)

Diferente de modelos de Deep Learning tradicionais, a Privacidade Diferencial sofre com a "Maldição da Dimensionalidade" de forma mais severa devido à injeção de ruído.

### Ação Necessária:

* **Eliminação de Redundância:** Se duas colunas transmitem a mesma informação (ex: Código do Município e Nome do Município), mantenha apenas a de menor cardinalidade.
* **Remoção de Identificadores:** Chaves primárias, IDs, CPFs ou nomes próprios devem ser removidos. Geradores sintéticos não conseguem (e não devem) replicar valores únicos com alta fidelidade sob Privacidade Diferencial.

---

## 3. Discretização de Variáveis Contínuas

O algoritmo AIM é otimizado para dados categóricos. Variáveis numéricas contínuas (como valores monetários ou idades exatas) geram um espaço de busca infinito ou muito esparso.

### Ação Necessária:

* **Transformação em Ordinais:** Converta valores contínuos em faixas (bins). Por exemplo, em vez de usar o salário exato, utilize faixas salariais ou decis.
* **Tratamento de Datas:** Datas completas (DD/MM/AAAA) devem ser decompostas em componentes de interesse, como apenas o ano ou o trimestre.

---

## 4. Gestão do Parâmetro `max_cells`

O parâmetro `max_cells` define o limite superior de complexidade que o AIM tentará modelar.

### Ação Necessária:

* **Sincronização com o Dataset:** Se após o wrangling o dataset ainda possuir colunas de alta cardinalidade, o valor de `max_cells` deve ser aumentado para permitir que o modelo capture essas relações, sob o custo de maior tempo de processamento.
* **Amostragem:** Para datasets massivos, utilize uma amostra representativa (ex: 50.000 a 100.000 registros) durante o treinamento do modelo sintético para garantir que as iterações do AIM terminem em tempo viável.

---

## 5. Tratamento de Valores Nulos e Ausentes

Valores nulos possuem significado estatístico, mas podem aumentar a cardinalidade se tratados incorretamente.

### Ação Necessária:

* **Codificação Explícita:** Substitua valores nulos por um código específico (ex: "NULO" ou -1) para que o AIM os trate como uma categoria distinta na distribuição.
* **Remoção de Colunas Esparsas:** Colunas com mais de 90% de valores ausentes devem ser descartadas, pois o ruído necessário para protegê-las tornará o dado sintético irrelevante.
