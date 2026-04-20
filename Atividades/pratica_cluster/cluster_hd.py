# IMPORTS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# 1. Abrir os dados
dados = pd. read_csv('HousingData.csv', sep=',')
#print(dados.head(10))

# 2. Normalizar os dados
dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']

dados_num = dados_num.fillna(dados_num.mean()) # --- Tratamento de NaN: média
dados_cat = dados_cat.fillna(dados_cat.mode()[0]) # --- Tratamento de NaN: preenche com 0

# 2.1 Normalizar Dados Numéricos
scaler = MinMaxScaler() # --- Instanciar normalizador
normalizador = scaler.fit(dados_num) # --- Treinar normalizador

pickle.dump(normalizador, open('normalizador_hd.pkl', 'wb')) # --- Salvar o normalizador para uso posterior

dados_num_norm = normalizador.transform(dados_num) # --- Normalizar dados
#print(dados_num_norm)

# 2.2 Normalizar Dados Categóricos
dados_cat_norm = pd.get_dummies(dados_cat, prefix='CHAS', prefix_sep='_', dtype=int)
#print(dados_cat_norm)

# 3. Reagrupar os objetos normalizador em uma dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns) # --- Converte a matriz numérica em dataframe

dados_norm = dados_num_norm.join(dados_cat_norm) # --- Juntar os dados numéricos normalizados com os dados categóricos normalizados
#print(dados_norm.columns)

# 4. Hiperparametizar
distortions = []
K = range(1, dados.shape[0])

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    # --- Calcular e armazenar a distorção de cada treinamento
    distortions.append(
        sum(
            np.min(
                cdist(dados_norm,
                      cluster_model.cluster_centers_,
                      'euclidean'), axis=1)/dados_norm.shape[0]
        )
    )
#print(distortions)

# 5. Determinar o número ótimo de cluster para o modelo
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distances.append(numerador/denominador)
numero_cluster_otimo = (K[distances.index(np.max(distances))])

# 6. Treinar o modelo com o número ótimo
cluster_model = KMeans(n_clusters=numero_cluster_otimo, random_state=42).fit(dados_norm)

pickle.dump(cluster_model, open('cluster_hd.pkl', 'wb'))
#print(cluster_model)
