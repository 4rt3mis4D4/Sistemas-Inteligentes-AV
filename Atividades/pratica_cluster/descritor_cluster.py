# IMPORTS
import pickle
import pandas as pd

# 1. Abrir o modelo de clusters + normalizador numérico
cluster_model = pickle.load(open('cluster_hd.pkl', 'rb'))
normalizador = pickle.load(open('normalizador_hd.pkl', 'rb'))

# 2. Desnormalizar os centroides
columns_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV', 'CHAS_0.0', 'CHAS_1.0']

# 3. Converter os centroides em dataframe
dataframe = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_names)
#print(dataframe)

atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(
        dataframe[['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV']]),
                columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV'])
#print(atributos_num_desnorm.head(5))

# 4. Denormalizar as colunas codificadas com One-Hot-Encoder
class_dataframe = dataframe[['CHAS_0.0', 'CHAS_1.0']].round(0).astype(int)

class_dataframe.columns = ['Não', 'Sim'] # --- 0.0 Não fazem fronteira com o Charles River e 1.0 Sim, fazem fronteira com o Charles River

class_dataframe = pd.from_dummies(class_dataframe)
class_dataframe.columns=['CHAS']

#print(class_dataframe)

# 5. Juntar os dataframes: colunas numéricas desnormalizadas e a categórica desnormalizada 
cluster = atributos_num_desnorm.join(class_dataframe)
print(cluster)
