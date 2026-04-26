# NORMALIZAÇÃO: Numérica
# CODIFICAÇÃO: MinMaxScaler (transforma os dados para um intervalo comum, geralmente [0,1])
# VARIÁVEIS: Contínua (podem assumir qualquer valor em um intervalo) e Discretas (valores específicos e contáveis, num int)

# Imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np 

# Dados à serem normalizados
dados = np.array([[1500], [3000], [5500], [10000]])
#print(dados)

# Instanciando Normalizador
scaler = MinMaxScaler(feature_range=(0,1))

# Normalizando os dados
dados_norm = scaler.fit_transform(dados)
#print(dados_norm)

# Desnormalizar os dados
dados_desnorm = scaler.inverse_transform(dados_norm)
print(dados_desnorm)
