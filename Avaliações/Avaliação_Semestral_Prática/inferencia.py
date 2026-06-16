# IMPORTS
import pandas as pd
from pickle import load
import numpy as np

# 1. DECLARANDO NOVOS DADOS
# -- Lista de Colunas do dataset
colunas = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
# -- Declarando novos dados
dados_novos = ['x', 's', 'n', 't', 'p', 'f', 'c', 
                'n', 'k', 'e', 'e', 's', 's', 'w', 
                'w', 'w', 'o', 'p', 'k', 's', 'u']

# -- Criando o DataFrame
dados_dataframe = pd.DataFrame([dados_novos], columns=colunas)

# 2. CARREGAR NORMALIZADORES + MODELO
normalizador_cat_ord = load(open('normalizador_cat_ord.pkl', 'rb'))
normalizador_cat_nom = load(open('normalizador_cat_nom.pkl', 'rb'))
modelo = load(open('modelo_rf_mushroom.pkl', 'rb'))

# 3. PRÉ-PROCESSAMENTO
# -- Ordinal
dados_ord_norm = dados_dataframe[['ring-number', 'gill-spacing', 'gill-size']].copy()
for col in ['ring-number', 'gill-spacing', 'gill-size']:
    encoder = normalizador_cat_ord[col]
    dados_ord_norm[col] = encoder.transform(dados_dataframe[col])

# -- Nominais
dados_nom_temp = pd.get_dummies(dados_dataframe.drop(columns=['ring-number', 'gill-spacing', 'gill-size']), dtype=int)

# Alinhamento
dados_nom_final = pd.DataFrame(0, index=dados_dataframe.index, columns=normalizador_cat_nom)
for col in dados_nom_temp.columns:
    if col in dados_nom_final.columns:
        dados_nom_final[col] = dados_nom_temp[col]

# -- Concatenação
X_inferencia = pd.concat([dados_ord_norm, dados_nom_final], axis=1)

# 4. PREDIÇÃO
predicao = modelo.predict(X_inferencia)
prob = modelo.predict_proba(X_inferencia).max()

resultado = 'Comestível (e)' if predicao[0] == 'e' else 'Venenoso (p)'
print(f"O cogumelo foi classificado como: {resultado}")
print(f"Confiança do modelo: {prob * 100:.2f}%")
