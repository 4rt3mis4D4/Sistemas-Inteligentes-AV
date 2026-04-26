# 🔢 Normalização de Dados (Numéricos e Categóricos)

Este diretório apresenta exemplos práticos de **normalização de dados**, abordando tanto variáveis **numéricas** quanto **categóricas**.

O objetivo é demonstrar como preparar dados corretamente para algoritmos de Machine Learning, garantindo consistência, desempenho e interpretabilidade.

---

## 🧠 1. Conceito Geral

### 🔹 O que é Normalização?
Normalização é o processo de transformar os dados para uma escala comum, evitando que atributos com magnitudes diferentes impactem negativamente o modelo.

### 🔹 Por que é importante?
- Melhora a performance dos algoritmos  
- Evita viés em cálculos de distância  
- Garante consistência entre treino e inferência  

### 🔹 Tipos abordados neste diretório:
- Dados **Numéricos**
- Dados **Categóricos Ordinais**
- Dados **Categóricos Nominais**

---

## 📁 2. Estrutura dos Arquivos

### 🔢 `numerica.py` — Normalização Numérica

Aplica normalização em dados numéricos utilizando **MinMaxScaler**.

**Passo a passo:**

1. **Definição dos Dados**
   - Vetor com valores numéricos (ex: salários)

2. **Instanciação**
   - Criação do `MinMaxScaler` com intervalo `[0,1]`

3. **Normalização**
   - Transforma os dados para a mesma escala

4. **Desnormalização**
   - Retorna os dados para a escala original

📌 **Tipo de variável:**
- Contínua (valores reais)
- Discreta (valores inteiros)

---

### 🔠 `categorica_ordinal.py` — Codificação Categórica Ordinal

Transforma categorias com **ordem natural** em valores numéricos.

**Técnica utilizada:** `LabelEncoder`

**Passo a passo:**

1. **Definição das Categorias**
   - Exemplo: fundamental → médio → superior

2. **Instanciação**
   - Criação do encoder

3. **Normalização**
   - Cada categoria recebe um número inteiro

4. **Desnormalização**
   - Conversão de volta para o valor original

📌 **Tipo de variável:**
- Ordinal (possui hierarquia)

⚠️ **Atenção:**
A ordem dos valores impacta diretamente o modelo.

---

### 🧩 `categorica_nominais.py` — Codificação Categórica Nominal

Transforma categorias **sem ordem** em representações numéricas.

**Técnica utilizada:** One-Hot Encoding

**Passo a passo:**

1. **Definição dos Dados**
   - Exemplo: cores (Vermelho, Azul, Verde)

2. **Normalização**
   - Criação de colunas binárias (0 ou 1)

3. **Desnormalização**
   - Reconstrução dos dados originais

📌 **Tipo de variável:**
- Nominal (sem hierarquia)

✔️ Evita criação de relações falsas entre categorias

---

## 🔑 3. Palavras-Chave

- Normalização  
- MinMaxScaler  
- LabelEncoder  
- One-Hot Encoding  
- Variáveis Numéricas  
- Variáveis Ordinais  
- Variáveis Nominais  
- Pré-processamento  

---

## 📌 Observações Finais

Este diretório demonstra conceitos fundamentais de **pré-processamento de dados**, etapa essencial em qualquer pipeline de Machine Learning.

Boas práticas destacadas:
- Escolha correta da técnica conforme o tipo de variável  
- Separação clara por tipo de dado  
- Possibilidade de reverter transformações (interpretabilidade)  
