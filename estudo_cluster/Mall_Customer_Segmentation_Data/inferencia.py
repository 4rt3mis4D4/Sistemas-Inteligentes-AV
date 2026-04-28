# Imports
import pandas as pd
import pickle
import numpy as np

# 1. Entrada de dados (recebe valores do novo dado)

# -- Salvando nomes das colunas
columns_name = ['CustomerID',
                'Gender', 
                'Age', 
                'Annual Income (k$)', 
                'Spending Score (1-100)']

# -- Declaração do novo dado
novo_cliente = [[201, 'Male', 32, 75, 82]]

# -- Cria um dataframe com os dados novos e estrutura desejada
cliente_dataframe = pd.DataFrame(novo_cliente, columns=columns_name)

# 2. Pré-Processamento (aplica a normalização salva anteriormente)

# -- Carregamento dos arquivos (normalizador e modelo treinado)
normalizador = pickle.load(open('normalizador.pkl', 'rb')) # -- Abre arquivo normalizador numérico
normalizador_cat = pickle.load(open('normalizador_cat.pkl', 'rb')) # -- Abre arquivo normalizador categórico
cluster = pickle.load(open('cluster.pkl', 'rb')) # -- Abre arquivo do modelo treinado

# -- Separar dado categórico do numérico
cliente_num = cliente_dataframe.drop(columns=['Gender'])
cliente_cat = cliente_dataframe['Gender']

# -- Normalizador numérico
cliente_num_norm = normalizador.transform(cliente_num)
cliente_cat_norm = pd.get_dummies(cliente_dataframe[['Gender']], prefix='Gender', prefix_sep='_', dtype=int)

# -- Declarando todas as colunas categóricas de normalização
colunas_treino = ['Female', 'Male']

# -- Tratamento das colunas categóricas
for col in colunas_treino:
    if col not in cliente_cat_norm.columns:
        cliente_cat_norm[col] = 0  # -- Adiciona a coluna faltando com valor zero

# -- Reordenar as colunas categóricas para garantir consistência
cliente_cat_norm = cliente_cat_norm[colunas_treino]

# -- Converter dados numéricos do cliente para dataframe
cliente_num_norm = pd.DataFrame(cliente_num_norm, columns= cliente_num.columns)

# -- Junta dados categóricos do cliente com os numéricos
cliente_novo_norm = cliente_num_norm.join(cliente_cat_norm)
#print(cliente_novo_norm.columns) # -- Exibe colunas do dataframe do novo cliente normalizado

# 3. Predição (recebe os novos dados e identifica qual cluster possui proximidade)
cluster_cliente = cluster.predict(cliente_novo_norm)
print(cluster_cliente)
