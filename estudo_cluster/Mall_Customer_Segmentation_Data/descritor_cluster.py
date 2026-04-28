# Imports
import pickle
import pandas as pd

# 1. Carregamento (abre arquivos: normalizador e modelo treinado)

cluster_model = pickle.load(open('cluster.pkl', 'rb')) # -- Abre modelo treinado
normalizador = pickle.load(open('normalizador.pkl', 'rb')) # -- Abre o normalizador

# 2. Desnormalização (converte os centroides de volta para a escala original)

# -- Salvando nomes das colunas
columns_name = ['CustomerID', 
                'Age', 
                'Annual Income (k$)', 
                'Spending Score (1-100)',
                'Female', 
                'Male']

# -- Converte os centroides em dataframe
dataframe = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_name)
#print(dataframe) # -- Exibindo dataframe de centroides

# -- Desnormalizar dados numéricos normalizados anteriormente
atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(
        dataframe[
        ['CustomerID', 
        'Age', 
        'Annual Income (k$)', 
        'Spending Score (1-100)']]),
        columns=['CustomerID', 
                'Age', 
                'Annual Income (k$)', 
                'Spending Score (1-100)'])
#print(atributos_num_desnorm.head(5)) # -- Exibindo dados desnormalizados

# 3. Tradução (converte as colunas categóricas de volta para o seu nome original)

# -- Limpeza das categorias
class_dataframe = dataframe[['Female', 'Male']].round(0).astype(int)

# -- Desnormalizando dados categóricos normalizados anteriormente
class_dataframe = pd.from_dummies(class_dataframe)
class_dataframe.columns=['Gender']
#print(class_dataframe) # -- Exibindo dados desnormalizados

# 4. Análise (exibe as caracteristicas médias de cada cluster criado)
cluster = atributos_num_desnorm.join(class_dataframe) # -- Junta as colunas desnormalizadas categóricas com as numéricas
print(cluster) # -- Exibindo cluster
