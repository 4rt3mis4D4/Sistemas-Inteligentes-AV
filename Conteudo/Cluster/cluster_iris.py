# SISTEMAS INTELIGENTES
# MODELOS NÃO SUPERVISIONADOS
# BASE IRIS

# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans # KMeans é um clusterizador 
import math
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist # método para cálculo de distâncias cartesianas
import numpy as np

# 1. Abrir os dados
dados = pd.read_csv('iris.csv', sep=';')
#print(dados.head(10))

# 2. Normalizar os dados
# 2.1 Separar atributos numéricos e atributos categóricos
dados_num = dados.drop(columns=['class'])
dados_cat = dados['class']

# 2.2 Normalizar os dados númericos
# -- Instanciar o normalizador
scaler = MinMaxScaler()

# -- Treinar o normalizador
normalizador = scaler.fit(dados_num)

# -- Salvar o normalizador para uso posterior
pickle.dump(normalizador, open('normalizador_iris.pkl', 'wb'))

# -- Normalizar os dados
dados_num_norm = normalizador.fit_transform(dados_num)
#print(dados_num_norm)

# 2.3 Normalizar os dados categóricos
dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)
#print(dados_cat_norm)

# 2.4 Reagrupar os objetos normalizados em uma dataframe 
# -- Converter a matriz numérica (dados_num_norm) em um dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)

# -- Juntar o dados_num_norms com o dado_cat_norm
dados_norm = dados_num_norm.join(dados_cat_norm)
#print(dados_norm)

# 3. HIPERPARAMETIZAR
# Vamos determinar o número ótimo de clusters antes do treinamento
distortions=[] # Matriz para armazernar as distoções
K = range(1, dados.shape[0])
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    

