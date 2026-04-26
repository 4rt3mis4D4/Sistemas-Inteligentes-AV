# NORMALIZAÇÃO: Categórica
# CODIFICAÇÃO: One-Hot Encoding (cria uma coluna binária (0 ou 1) para cada categoria, evitando a criação de ordens artificiais)
# VARIÁVEIS: Nominais (sem ordem ou hierarquia)

# Imports
import pandas as pd

# Dados categóricos para serem normalizados
df = pd.DataFrame({'cor': ['Vermelho', 'Azul', 'Verde']})
#print(df)

# Normalização dos dados categóricos
df_norm = pd.get_dummies(df, columns=['cor'], prefix='cor', dtype=int)
#print(df_norm)

# Desnormalização dos dados categóricos ja normalizados
df_desnorm = pd.from_dummies(df_norm, sep='_')
print(df_desnorm)
