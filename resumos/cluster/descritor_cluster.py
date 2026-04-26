# MODELO NÃO SUPERVISIONADO - BASE IRIS
# Objetivo de "descritor_cluster.py": torna o resultado (número normalizados) compreensível para humanos

# Imports
import pickle
import pandas as pd

# 1. CARREGAMENTO (recupera modelo treinado e normalizador)
# =========================================================
cluster_model = pickle.load(open('cluster_iris.pkl', 'rb')) # -- Abrir modelo de clusters
normalizador = pickle.load(open('normalizador_iris.pkl', 'rb')) # -- Abrir normalizador numérico

# 2. DESNORMALIZAÇÃO (converte os centroides de volta para a escala original)
# ===========================================================================

# -- Desnormalizar os Centroides
columns_name = ['sepal_length', 
                'sepal_width', 
                'petal_length', 
                'petal_width',
                'Iris-setosa', 
                'Iris-versicolor', 
                'Iris-virginica']

# -- Converter os centroides em dataframe
dataframe = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_name)
#print(dataframe)

# -- Desnormalizar dados numéricos
atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(
        dataframe[
            ['sepal_length', 
             'sepal_width', 
             'petal_length', 
             'petal_width']
        ]),
        columns = ['sepal_length', 
             'sepal_width', 
             'petal_length', 
             'petal_width'])
#print(atributos_num_desnorm.head(5))

# 3. TRADUÇÃO (converte as colunas categóricas de volta para o seu nome original)
# ===============================================================================

# --- Desnormalizar dados categóricos
class_dataframe = dataframe[['Iris-setosa', 
                            'Iris-versicolor', 
                            'Iris-virginica']].round(0).astype(int)

class_dataframe = pd.from_dummies(class_dataframe)
class_dataframe.columns=['class']
#print(class_dataframe)

# 4. ANÁLISE (Exibe as características médias de cada cluster criado)
# ===================================================================
cluster = atributos_num_desnorm.join(class_dataframe) # -- Juntar colunas desnormalizadas numéricas e categóricas
print(cluster)
