from sklearn.preprocessing import LabelEncoder

# Dados categóricos para normalizar
cores = ['Vermelho', 'Azul', 'Verde', 'Azul']

# Construir o codificador
encoder = LabelEncoder() # Mapeamento - Converte variáveis categóricas em números inteiros

cores_normalizadas = encoder.fit_transform(cores) # Aprende e aplica o mapeamento

print('Cores Naturais:')
print(cores)

print('\nCores Normalizadas:')
print(cores_normalizadas)


print('\nClasses Codificadas:')
print(encoder.classes_) # Substitui cada cor pelo indíce

# Inverte uma cor codificada
print('\nCor Natural:', encoder.classes_[0])
