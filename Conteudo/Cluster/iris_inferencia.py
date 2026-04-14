import pandas as pd
import pickle

# Dados para criar os data frames
columns_names = ['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width',
                 'Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica']

# Cria um dataframe vazio, com a estrutura desejada
flor_dataframe = pd.DataFrame(columns = columns_names) 

# Nova flor
nova_flor = [[6.4, 2.8, 5.6, 2.1]] # Medida da pétala e do cálice

# Abrir o normalizador
normalizador = pickle.load(open('normalizador_iris.pkl', 'rb'))

# Abrir o modelo salvo
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

# Normalizar os dados de entrada
nova_flor_norm = normalizador.transform(nova_flor)
#print(nova_flor_normalizada)

# Converter a nova instancia normalizada em dataframe
nova_flor_norm = pd.DataFrame(nova_flor_norm, 
                            columns = ['sepal_length', 
                                        'sepal_width', 
                                        'petal_length', 
                                        'petal_width'])

# Concatenar o DataFrame da nova instância com o DataFrame vazio (que possui o formato final do objeto)
flor_nova_instancia = pd.concat([nova_flor_norm, flor_dataframe]).fillna(0)
#print(flor_nova_instancia)

cluster_flor = cluster_iris.predict(flor_nova_instancia)
print('Cluster da nova flor: ', cluster_flor)