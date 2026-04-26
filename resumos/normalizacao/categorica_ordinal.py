# NORMALIZAÇÃO: Categórica
# CODIFICAÇÃO: Label Encoder (Atribui um número inteiro para cada categoria)
# VARIÁVEIS: Ordinais (possuem uma ordem natural)

# Imports
from sklearn.preprocessing import LabelEncoder

# Categorias para normalizar
categorias = ['fundamental', 'médio', 'superior']
#print(categorias)

# Instanciar normalizador
encoder = LabelEncoder()

# Normalizar os dados categóricos
cat_norm = encoder.fit_transform(categorias)
#print(cat_norm)

# Denormalizar dados categóricos ja normalizados
cat_desnorm = encoder.inverse_transform(cat_norm)
print(cat_desnorm)
