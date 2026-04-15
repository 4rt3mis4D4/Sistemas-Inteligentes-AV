# Imports
import pickle
import pandas as pd

# 1. Abrir o modelo de clusters + normalizador
cluster_model = pickle.load(open('cluster_hd.pkl', 'rb'))

normalizador = pickle.load(open('normalizador_hd.pkl', 'rb'))

# 2. Desnormalizar os centroides + converter em dataframe
columns_names = ['CRIM',
                 'ZN',
                 'INDUS',
                 'NOX',
                 'RM',
                 'AGE',
                 'DIS',
                 'RAD',
                 'TAX',
                 'PTRATIO',
                 'B',
                 'LSTAT',
                 'MEDV',
                 'CHAS_0',
                 'CHAS_1']

dataframe = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_names)
print(dataframe)