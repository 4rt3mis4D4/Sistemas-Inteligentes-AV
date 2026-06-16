# 🧠 Guia Prático de Machine Learning com Classificação

> Este projeto demonstra a construção completa de um modelo de **Machine Learning supervisionado para classificação**, abrangendo desde o pré-processamento dos dados até a avaliação final do modelo.

---

# 🎯 Objetivo

O objetivo é desenvolver um modelo capaz de identificar a qual classe uma determinada observação pertence com base em um conjunto de atributos disponíveis.

## 📖 O que é um Classificador?

A classificação é uma das principais tarefas de Aprendizado de Máquina Supervisionado. Nela, o modelo aprende padrões a partir de dados previamente rotulados para prever a categoria de novas observações.

Durante o treinamento, o algoritmo analisa exemplos históricos e aprende relações entre as variáveis de entrada e a variável alvo, tornando-se capaz de realizar previsões sobre dados nunca vistos anteriormente.

---

# 🚀 Pipeline de Desenvolvimento

O projeto segue um fluxo estruturado de desenvolvimento, composto pelas etapas descritas abaixo.

---

## 📂 1. Carregamento dos Dados

A primeira etapa consiste na importação do conjunto de dados para o ambiente de desenvolvimento.

Nessa fase, os registros são carregados e preparados para as etapas subsequentes de análise e processamento.

---

## 🧹 2. Remoção de Variáveis Não Relevantes

Nem todas as variáveis contribuem para o aprendizado do modelo.

Informações excessivamente específicas, identificadores únicos ou atributos que não possuem capacidade de generalização podem introduzir ruído e prejudicar o desempenho da classificação.

Por esse motivo, é realizada uma etapa de seleção e limpeza das variáveis utilizadas.

---

## 🗂️ 3. Identificação dos Tipos de Variáveis

Antes do treinamento, é necessário compreender a natureza de cada atributo presente no conjunto de dados.

As variáveis podem ser classificadas em:

### 🔢 Variáveis Numéricas

Representam valores quantitativos.

### 📊 Variáveis Categóricas Ordinais

Possuem uma ordem ou hierarquia natural entre suas categorias.

### 🏷️ Variáveis Categóricas Nominais

Representam categorias sem qualquer relação de ordem.

### 🎯 Variável Alvo

Corresponde à informação que o modelo deverá prever.

---

## ✂️ 4. Separação entre Atributos e Variável Alvo

Os dados são divididos em dois grupos:

* **Atributos (Features):** informações utilizadas para realizar as previsões.
* **Variável Alvo (Target):** categoria que o modelo deve aprender a identificar.

Posteriormente, o conjunto é separado em subconjuntos de treinamento e teste, permitindo avaliar a capacidade de generalização do modelo.

---

## ⚖️ 5. Pré-processamento e Normalização

Os algoritmos de Machine Learning geralmente exigem que os dados estejam em formatos adequados para processamento.

Durante essa etapa são realizadas transformações como:

* Escalonamento de variáveis numéricas;
* Conversão de variáveis categóricas em representações numéricas;
* Padronização dos dados de entrada;
* Armazenamento das transformações para uso futuro.

Essas operações garantem consistência e melhor desempenho durante o treinamento.

---

## ⚖️ 6. Balanceamento das Classes

Em muitos conjuntos de dados, algumas classes possuem quantidade significativamente maior de exemplos do que outras.

Esse desbalanceamento pode levar o modelo a favorecer as classes mais frequentes.

Para minimizar esse problema, são aplicadas técnicas de balanceamento que tornam a distribuição das classes mais uniforme, contribuindo para previsões mais equilibradas.

---

## 🎛️ 7. Otimização de Hiperparâmetros

Os algoritmos possuem configurações internas conhecidas como hiperparâmetros.

A escolha adequada desses parâmetros pode impactar diretamente o desempenho do modelo.

Por isso, são utilizados métodos de busca automatizada para identificar combinações capazes de produzir melhores resultados.

---

## 🏋️ 8. Treinamento do Modelo

Após a preparação dos dados e definição das melhores configurações, o algoritmo é treinado.

Durante esse processo, o modelo aprende padrões e relações existentes entre os atributos e a variável alvo, construindo sua capacidade de realizar previsões.

Ao final, o modelo treinado é armazenado para utilização futura.

---

## 🔄 9. Validação Cruzada

A validação cruzada é utilizada para avaliar a robustez do modelo.

Nessa técnica, os dados são divididos em múltiplas partições, permitindo que diferentes subconjuntos sejam utilizados alternadamente para treinamento e validação.

Esse procedimento fornece estimativas mais confiáveis do desempenho do modelo.

As principais métricas analisadas incluem:

* ✅ Acurácia
* ✅ Precisão
* ✅ Recall
* ✅ F1-Score

---

## 📊 10. Avaliação Final

Após o treinamento, o modelo é submetido a um conjunto de dados que não participou do processo de aprendizagem.

Essa etapa permite medir sua capacidade real de generalização e identificar seu desempenho em situações práticas.

---

## 🧩 11. Matriz de Confusão

A matriz de confusão oferece uma visão detalhada dos acertos e erros realizados pelo classificador.

Por meio dela, é possível verificar:

* Quantas observações foram classificadas corretamente;
* Quais classes apresentam maior dificuldade de identificação;
* Quais tipos de erro ocorrem com maior frequência.

A visualização normalmente é apresentada por meio de gráficos que facilitam a interpretação dos resultados.

---

## 🔬 12. Métricas Avançadas de Desempenho

Para uma análise cirúrgica do desempenho classe por classe, extraímos as seguintes métricas da Matriz de Confusão:



* **Sensibilidade (Recall):** Dos que *realmente eram* da Classe A, quantos o modelo conseguiu identificar?

    $$Sensibilidade = \frac{VP}{VP + FN}$$

* **Especificidade:** Dos que *NÃO eram* da Classe A, quantos o modelo corretamente descartou?

    $$Especificidade = \frac{VN}{VN + FP}$$



*(Onde VP = Verdadeiros Positivos, VN = Verdadeiros Negativos, FP = Falsos Positivos, FN = Falsos Negativos)*



No final, calculamos as médias globais dessas métricas para entender a estabilidade geral do nosso classificador.

---

# 🛠️ Tecnologias Utilizadas

| Tecnologia          | Finalidade                                 |
| ------------------- | ------------------------------------------ |
| 🐍 Python           | Desenvolvimento da solução                 |
| 🐼 Pandas           | Manipulação e processamento de dados       |
| 🔢 NumPy            | Operações matemáticas e numéricas          |
| 🤖 Scikit-Learn     | Pré-processamento, treinamento e avaliação |
| ⚖️ Imbalanced-Learn | Técnicas de balanceamento                  |
| 📈 Matplotlib       | Visualização de resultados                 |
