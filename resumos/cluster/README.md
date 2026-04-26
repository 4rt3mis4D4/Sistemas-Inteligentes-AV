# 📊 Clustering com K-Means (Dataset Iris)

Este diretório apresenta uma implementação completa da técnica de **AGRUPAMENTO (CLUSTERING)** utilizando o algoritmo **K-Means**.

O objetivo é demonstrar, de forma prática e organizada, como funciona um pipeline de Machine Learning não supervisionado.

---

## 🧠 1. Conceito Geral

### 🔹 O que é Clustering?

Clustering é uma técnica de **aprendizado não supervisionado** que agrupa dados semelhantes em conjuntos chamados **clusters**. Diferente da classificação, não existem rótulos prévios — o modelo identifica padrões ocultos nos dados.

### 🔹 Objetivo

Organizar os dados de forma que elementos dentro do mesmo grupo sejam mais semelhantes entre si do que em relação a outros grupos.

### 🔹 Aplicação neste projeto

Identificar padrões entre diferentes tipos de flores com base em suas características físicas (comprimento e largura de pétala e cálice).

---

## 📁 2. Estrutura dos Arquivos

### 🧩 `cluster.py` — Treinamento e Hiperparametrização

Responsável pelo processo de aprendizado do modelo.

**Passo a passo:**

1. **Tratamento de Dados**

   * Leitura do dataset `.csv`
   * Aplicação de **normalização (MinMaxScaler)**

2. **Hiperparametrização**

   * Uso do método do **cotovelo (Elbow Method)**
   * Cálculo da **distorção**
   * Definição do **número ótimo de clusters**

3. **Treinamento**

   * Instanciação do **K-Means**
   * Ajuste do modelo aos dados

4. **Persistência**

   * Salvamento do modelo e do scaler em `.pkl`
   * Uso de **Pickle**

---

### 🔍 `descritor_cluster.py` — Interpretação dos Resultados

Responsável por tornar os dados compreensíveis.

**Passo a passo:**

1. **Carregamento**

   * Importa modelo e scaler

2. **Desnormalização**

   * Converte **centroides** para escala original

3. **Tradução**

   * Reconverte dados categóricos
   * Associa clusters às espécies

4. **Análise**

   * Exibe características médias dos clusters

---

### 🚀 `inferencia.py` — Uso Prático

Simula o uso em produção.

**Passo a passo:**

1. **Entrada de Dados**

   * Recebe uma nova amostra

2. **Pré-processamento**

   * Aplica a mesma normalização

3. **Predição**

   * Executa `predict`
   * Calcula cluster mais próximo

4. **Resultado**

   * Retorna o cluster correspondente

---

## 🔑 3. Palavras-Chave

* Modelo Não Supervisionado
* K-Means
* Normalização
* Centroides
* Pickle
* Número Ótimo de Clusters
* Distância Euclidiana
* Inferência

---

## 📌 Observações Finais

Este projeto demonstra um fluxo completo de Machine Learning, incluindo:

* Pré-processamento de dados
* Treinamento de modelo
* Persistência
* Inferência
* Interpretação de resultados

Ideal para estudos em **Sistemas Inteligentes** e **Ciência de Dados**.

---

Se ainda ficar estranho no GitHub, me fala que eu ajusto exatamente pro teu tema (dark/light) ou deixo mais “visual profissional estilo portfólio”.
