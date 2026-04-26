# MODELO NÃO SUPERVISIONADO - BASE IRIS
# Objetivo de "inferencia.py": recebe novos dados e determina à qual grupo pertence

# Imporst
import pandas as pd
import pickle

# 1. ENTRADA DE DADOS (recebe os valores do novo dado)
# ====================================================

# -- Dados para criar o dataframe
columns_name = ['sepal_length', 
                'sepal_width', 
                'petal_length', 
                'petal_width',
                'Iris-setosa', 
                'Iris-versicolor', 
                'Iris-virginica']

# -- Cria um dataframe vazio, com a estrutura desejada
flor_dataframe = pd.DataFrame(columns=columns_name)

# -- Novo dado (flor): medida da pétala e do cálice
nova_flor = [[6.4, 2.8, 5.6, 2.1]]

# 2. PRÉ-PROCESSAMENTO (aplica a mesma normalização)
# ==================================================
normalizador = pickle.load(open('normalizador_iris.pkl', 'rb')) # -- Abrir o normalizador salvo
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb')) # -- Abrir o modelo salvo

nova_flor_norm = normalizador.transform(nova_flor) # -- Normalizar os dados de entrada
#print(nova_flor_norm)

# -- Converter a nova instancia normalizada em dataframe
nova_flor_norm = pd.DataFrame(nova_flor_norm,
                              columns=['sepal_length', 
                                        'sepal_width', 
                                        'petal_length', 
                                        'petal_width'])

# -- Concatenar o DataFrame da nova instancia com o DataFrame vazio (que possui o formato final do objeto)
flor_nova_instancia = pd.concat([nova_flor_norm, flor_dataframe]).fillna(0)
#print(flor_nova_instancia)

# 3. PREDIÇÃO (recebe os novos dados e identifica qual cluster possui proximidade)
# ================================================================================
cluster_flor = cluster_iris.predict(flor_nova_instancia)

# 4. RESULTADO (informa o cluster correspondente ao novo dado)
# ============================================================
print('Cluster da nova flor: ', cluster_flor)
