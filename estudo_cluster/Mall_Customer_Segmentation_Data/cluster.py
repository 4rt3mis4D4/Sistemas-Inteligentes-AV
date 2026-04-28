# Imports
from sklearn.preprocessing import MinMaxScaler # Normalizador numérico
import pandas as pd # Normaliazr dados categóricos (Nominais: One-Hot Encoder)
import pickle # Salvar dados normalizados
from sklearn.cluster import KMeans # Método de aprendizado cluster
import numpy as np # Para cálculos e processamento de dados
from scipy.spatial.distance import cdist # Cálculo da distância
import math # Para funções matemáticas

# 1. Tratamento de Dados (abre o .CSV e realiza a normalização)

dados = pd.read_csv('Mall_Customers.csv', sep=',') # -- Abrir dataset
#print(dados.head(10)) # -- Visualizar dados do dataset

# -- Separa atributos categóricos dos atributos numéricos
dados_num = dados.drop(columns=['Gender'])
dados_cat = dados['Gender']

# -- Instanciar normalizador numérico
scaler = MinMaxScaler(feature_range=(0,1))

# -- Treinando normalizador
normalizador = scaler.fit(dados_num)

# -- Salvando normalizador treinado para uso posterior
pickle.dump(normalizador, open('normalizador.pkl', 'wb'))

# -- Normalizando dados numéricos
dados_num_norm = normalizador.fit_transform(dados_num)
#print(dados_num_norm) # -- Exibindo dados normalizados

# -- Normalizando dados categóricos
dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

# -- Armazena lista de colunas categóricas normalizadas
colunas_categoricas = dados_cat_norm.columns.to_list()

# -- Salvando normalizador categórico treinado para uso posterior
pickle.dump(colunas_categoricas, open('normalizador_cat.pkl', 'wb'))

# -- Converter dados numéricos para dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns= dados_num.columns)

# -- Juntar dados numéricos com categóricos
dados_norm = dados_num_norm.join(dados_cat_norm)
#print(dados_norm.columns) # -- Exibindo colunas com dados ja normalizados

# 2. Hiperparametizar (cálculo das distorções + determinar número ótimo de clusters)

# -- Cálculo das distorções
distortions = []
K = range(1, dados.shape[0])
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    # -- Calcular e armazenar distorções de cada treinamento
    distortions.append(
        sum(
            np.min(
                cdist(dados_norm,
                   cluster_model.cluster_centers_, 'euclidean'), axis=1)/dados_norm.shape[0]
        )
    )
#print(distortions) # -- Exibindo distorções

# -- Determinar o número ótimo de cluster para o modelo
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs (
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*y0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distances.append(numerador/denominador)
numero_otimo_cluster = (K[distances.index(np.max(distances))])

# 3. Treinamento (Treina o modelo com número ótimo de cluster)
cluster_model = KMeans(n_clusters=numero_otimo_cluster, random_state=42).fit(dados_norm)

# 4. Persistências (Salva o modelo para uso posterior)
pickle.dump(cluster_model, open('cluster.pkl', 'wb'))
