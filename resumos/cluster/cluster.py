# MODELO NÃO SUPERVISIONADO - BASE IRIS
# Objetivo de "cluster.py": O modelo aprende a agrupar
 
# Imports 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# 1. TRATAMENTO DE DADOS (abre o .CSV e realiza a Normalização)
# =============================================================
dados = pd.read_csv('iris.csv', sep=';') # -- Abrir os dados
#print(dados.head(10))

# -- Separa atributos numéricos e atributos categóricos
dados_num = dados.drop(columns=['class'])
dados_cat = dados['class']

scaler = MinMaxScaler() # -- Instanciar NORMALIZADOR NUMÉRICO

normalizador = scaler.fit(dados_num) # --- Treinar o normalizador

pickle.dump(normalizador, open('normalizador_iris.pkl', 'wb')) # --- Salvar normalizador para uso posterior

dados_num_norm = normalizador.fit_transform(dados_num) # -- Normalizar os dados NUMÉRICOS
#print(dados_num_norm)

dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int) # -- Normalizar os dados CATEGÓRICOS: One Hot Encoding

dados_num_norm = pd.DataFrame(dados_num_norm, columns= dados_num.columns) # -- Converter a matriz numérica em um dataframe
dados_norm = dados_num_norm.join(dados_cat_norm) # -- Juntar o dados normalizados numéricos com categóricos
#print(dados_norm.columns)

# 2. HIPERPARAMETIZAR (cálculo de Distorção e encontrar número ótimo de Clusters)
# ===============================================================================
distortions = [] # -- Matriz para armazenar distorções
K = range(1, dados.shape[0])
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    # -- Calcular e armazenar a distorção de cada treinamento
    distortions.append(
        sum(
            np.min(
                cdist(dados_norm,
                      cluster_model.cluster_centers_,
                    'euclidean'), axis=1)/dados_norm.shape[0]
        )
    )
#print(distortions)

# -- Determinar o número ótimo de cluister para o modelo
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs (
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distances.append(numerador/denominador)
numero_cluster_otimo = (K[distances.index(np.max(distances))])

# 3. TREINAMENTO (Treinar modelo KMeans com o número ideal de clusters)
# =====================================================================
cluster_model = KMeans(
                        n_clusters=numero_cluster_otimo,
                        random_state=42).fit(dados_norm)

# 4. PERSISTÊNCIA (Salvar modelo para uso futuro)
# ===============================================
pickle.dump(cluster_model, open('cluster_iris.pkl', 'wb'))
