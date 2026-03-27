from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pickle 

# Dados numéricos
dado = np.array([
    [1500], [3000], [5500], [10000]
])

# Instanciar o normalizador 
scaler = MinMaxScaler()

# Treinar o modelo normalizador para uso posterior
scaler_model = scaler.fit(dado) # Método fit() treina o modelo normalizador

# Salvar o modelo normalizador para uso posterior
pickle.dump(scaler_model,open('scaler1.pkl', 'wb'))

# Normalizar os dados
dados_norm = scaler_model.fit_transform(dado)

print(dado)
print(dados_norm)
