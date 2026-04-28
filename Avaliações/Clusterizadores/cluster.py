# Imports
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.cluster import KMeans
import math
from scipy.spatial.distance import cdist
import numpy as np

# 1. Tratamento de dados (abre o .CSV e realiza a Normalização)
dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', sep=',') # -- Abrir dados
#print(dados.head(5)) # -- Exibe dados do .CSV

# -- Separa atributos numéricos e atributos categóricos
# Numéricos: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE 
dados_num = dados.drop(columns=['Gender', 
                                'family_history_with_overweight', 
                                'FAVC', 
                                'CAEC', 
                                'SMOKE', 
                                'SCC', 
                                'CALC', 
                                'MTRANS',
                                'NObeyesdad'])
# Categóricos Ordinais: CAEC, CALC, NObeyesdad
dados_cat_ord = dados[['CAEC', 'CALC', 'NObeyesdad']]

# Categóricos Nominais: Gender, family_history, FAVC, SMOKE, SCC, MTRANS
dados_cat_nom = dados [['Gender', 
                       'family_history_with_overweight', 
                       'FAVC', 
                       'SMOKE', 
                       'SCC', 
                       'MTRANS']]

# -- Normalização Numérica
scaler = MinMaxScaler() # -- Instanciando normalizador numérico
normalizador = scaler.fit(dados_num) # -- Treinando normalizador

pickle.dump(normalizador, open('normalizador.pkl', 'wb')) # -- Salvando normalizador para uso futuro

dados_num_norm = normalizador.transform(dados_num) # -- Normalizando dados numéricos
#print(dados_num) # -- Exibe dados numéricos normalizados

# -- Normalizar Categóricos Ordinais: LabelEncoder
encoder = LabelEncoder() # -- Instanciando normalizador

# -- Normalizando dados categóricos ordinais
dict_encoders = {} 
dados_cat_ord_copy = dados_cat_ord.copy()

for col in dados_cat_ord.columns:
    le = LabelEncoder()
    dados_cat_ord_copy[col] = le.fit_transform(dados_cat_ord[col])
    dict_encoders[col] = le 

dados_cat_ord_norm = dados_cat_ord_copy

pickle.dump(dict_encoders, open('normalizador_cat_ord.pkl', 'wb')) # -- Salvando normalizador para uso futuro

# -- Normalizar Categóricos Nominais: One-Hot Encoding 
dados_cat_nom_norm = pd.get_dummies(dados_cat_nom, prefix_sep='_', dtype=int)
colunas_categoricais_ordinais = dados_cat_nom_norm.columns.to_list() # -- Armazena colunas categóricas nominais

pickle.dump(dados_cat_nom_norm, open('normalizador_cat_nom.pkl', 'wb')) # -- Salvando normalizador para uso futuro

# -- Converte dados numéricos para dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)
#print(dados_num_norm)

dados_norm = pd.concat([dados_num_norm, dados_cat_nom_norm, dados_cat_ord_norm], axis=1)
#print(dados_norm.columns)

# 2. Hiperparametizar (cálculo das distorções + determinar número ótimo de clusters)

# -- Cálculo das distorções
distortions = []
K = range(1, 11)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_norm)

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
