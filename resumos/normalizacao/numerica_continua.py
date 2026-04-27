# NORMALIZAÇÃO: Numérica
# CODIFICAÇÃO: MinMaxScaler (transforma os dados em um intervalo comum (0,1))
# VARIÁVEL: Contínua (pode assumir qualquer valor em um intervalo)

# Imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Dados à serem normalizados
dados = np.array([[25,00], [142,00], [850,00], [1200,00]])
print(dados) # -- Exbindo dados originais

# Instanciando normalizador 
scaler = MinMaxScaler(feature_range=(0,1))

# Normalizando dados numéricos contínuos
dados_norm = scaler.fit_transform(dados)
print(dados_norm) # -- Exibindo dados normalizados

# Desnormalizar dados numérocos dados
dados_desnorm = scaler.inverse_transform(dados_norm)
print(dados_desnorm) # -- Exibindo dados desnormalizados (retorna para dados originais)
