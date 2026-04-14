# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# 1. Abrir os dados
dados = pd.read_csv('HousingData.csv', sep=',')
#print(dados.head(10))

dados = dados.fillna(dados.mean())

# 2. Normalizar os dados
# --- Separação dos atributos numéricos e categóricos 
dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']

# --- Normalizar Dados Numéricos ---
# Instanciar normalizador
scaler = MinMaxScaler()

# Treinar normalizador
normalizador = scaler.fit(dados_num)

# Salvar o normalizador para uso posterior
pickle.dump(normalizador, open('normalizador_hd.pkl', 'wb'))

# Normalizar os dados
dados_num_norm = normalizador.transform(dados_num)
#print(dados_num_norm)

# --- Normalizar Dados Categóricos ---
dados_cat_norm = pd.get_dummies(dados_cat, prefix='CHAS', prefix_sep='_', dtype=int)
#print(dados_cat_norm)

# 3. Reagrupar os objetos normalizador em uma dataframe
# --- Converter a matriz numérica em um dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

# --- Juntar os dados_num_norms com o dado_cat_norm
dados_norm = dados_num_norm.join(dados_cat_norm)
#print(dados_norm.columns)

# 4. Hiperparametizar 
# --- Determinar o número ótimo de clusters antes do treinamento
distortions=[]
K = range(1, dados.shape[0])

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    # calcular e armazenar a distorção de cada treinamento
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
numero_clusters_otimo = (K[distances.index(np.max(distances))])

# --- Treinar o modelo com o número ótimo
cluster_model = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(dados_norm)

# --- Salvar o modelo para uso posterior
pickle.dump(cluster_model, open('cluster_hd.pkl', 'wb'))
