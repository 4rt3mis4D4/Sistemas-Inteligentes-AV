```md
# 📊 Clustering com K-Means (Dataset Iris)

Este diretório apresenta uma implementação completa da técnica de **AGRUPAMENTO (CLUSTERING)** utilizando o algoritmo **K-Means**, aplicada ao clássico dataset Iris.

O objetivo é demonstrar, de forma prática e organizada, como funciona um pipeline de Machine Learning não supervisionado — desde o pré-processamento até a inferência.

---

## 🧠 1. Conceito Geral

### 🔹 O que é Clustering?
Clustering é uma técnica de **aprendizado não supervisionado** que agrupa dados semelhantes em conjuntos chamados **clusters**. Diferente da classificação, não existem rótulos prévios — o modelo identifica padrões ocultos nos dados.

### 🔹 Objetivo
Organizar os dados de forma que elementos dentro do mesmo grupo sejam mais semelhantes entre si do que em relação a outros grupos.

### 🔹 Aplicação neste projeto
Identificar padrões entre diferentes tipos de flores com base em suas características físicas (comprimento e largura de sépalas e pétalas).

---

## 📁 2. Estrutura dos Arquivos

### 🧩 `cluster.py` — Treinamento e Hiperparametrização
Responsável por todo o processo de aprendizado do modelo.

**Passo a passo:**
1. **Tratamento de Dados**
   - Leitura do dataset `.csv`
   - Aplicação de **normalização (MinMaxScaler)**

2. **Hiperparametrização**
   - Uso do método do **cotovelo (Elbow Method)**
   - Cálculo da **distorção** para diferentes valores de K
   - Definição do **número ótimo de clusters**

3. **Treinamento**
   - Instanciação do algoritmo **K-Means**
   - Ajuste do modelo aos dados

4. **Persistência**
   - Salvamento do modelo treinado e do scaler em arquivos `.pkl`
   - Uso de **Pickle** para reutilização futura

---

### 🔍 `descritor_cluster.py` — Interpretação dos Resultados
Traduz os resultados do modelo para algo compreensível.

**Passo a passo:**
1. **Carregamento**
   - Importa o modelo e o normalizador salvos

2. **Desnormalização**
   - Converte os **centroides** para a escala original (cm)

3. **Tradução**
   - Reconverte dados categóricos (One-Hot Encoding)
   - Associa clusters às espécies (Setosa, Versicolor, Virginica)

4. **Análise**
   - Exibe as características médias de cada cluster

---

### 🚀 `inferencia.py` — Uso em Produção
Simula o uso do modelo com novos dados.

**Passo a passo:**
1. **Entrada de Dados**
   - Recebe medidas de uma nova flor

2. **Pré-processamento**
   - Aplica a mesma **normalização** usada no treinamento

3. **Predição**
   - Executa o método `predict`
   - Identifica o cluster mais próximo via distância

4. **Resultado**
   - Retorna o cluster correspondente

---

## 🔑 3. Palavras-Chave

- Modelo Não Supervisionado  
- K-Means  
- Normalização  
- Centroides  
- Pickle (Serialização de modelos)  
- Número Ótimo de Clusters  
- Distância Euclidiana  
- Inferência  

---

## 📌 Observações Finais

Este projeto demonstra um fluxo completo de Machine Learning, destacando boas práticas como:
- Padronização de dados
- Separação de responsabilidades por arquivo
- Reutilização de modelos treinados
- Clareza na interpretação dos resultados

```
