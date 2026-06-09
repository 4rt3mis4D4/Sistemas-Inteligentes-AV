# IMPORTS
import pandas as pd
from pickle import load
import numpy as np

# 1. DECLARANDO DADOS NOVOS
columns_name = [
    'original_price', 'discount_pct', 'final_price', 'quantity', 
    'purchase_amount', 'is_weekend', 'is_black_friday',
    'gender', 'city', 'customer_segment', 'product_category', 'payment_method'
]

novos_dados = [[
    174.67,        # original_price
    25,            # discount_pct
    131.00,        # final_price
    1,             # quantity
    131.00,        # purchase_amount
    0,             # is_weekend
    0,             # is_black_friday
    'Male',        # gender (Nominal)
    'Phoenix',     # city (Nominal)
    'New',         # customer_segment (Nominal)
    'Groceries',   # product_category (Nominal)
    'Credit Card'  # payment_method (Nominal)
]]

# -- Cria um dataframe com os novos dados e estrutura de colunas
dados_dataframe = pd.DataFrame(novos_dados, columns=columns_name)

# 2. PRÉ-PROCESSAMENTO
dados_num = ['original_price', 'discount_pct', 'final_price', 'quantity', 
             'purchase_amount', 'is_weekend', 'is_black_friday']
dados_cat_nom = ['gender', 'city', 'customer_segment', 'product_category', 'payment_method']
pasta_modelo = 'age_group'

normalizador = load(open(f'{pasta_modelo}/normalizador_num.pkl', 'rb'))
normalizador_cat_nom = load(open(f'{pasta_modelo}/normalizador_cat_nom.pkl', 'rb'))
modelo = load(open(f'{pasta_modelo}/modelo_rf.pkl', 'rb'))

# 3. NORMALIZAÇÃO
# -- Numéricos
dados_num_norm = pd.DataFrame(normalizador.transform(dados_dataframe[dados_num]), columns=dados_num, index=dados_dataframe.index)

# -- Nominais
dados_nom_norm = pd.get_dummies(dados_dataframe[dados_cat_nom], prefix=dados_cat_nom, dtype=int)

# Alinhamento estrutural
dados_nom_final = pd.DataFrame(0, index=dados_dataframe.index, columns=normalizador_cat_nom)
for col in normalizador_cat_nom:
    if col in dados_nom_norm.columns:
        dados_nom_final[col] = dados_nom_norm[col]

# -- Concatenação
dados_concat = pd.concat([dados_num_norm, dados_nom_final], axis=1)

# 3. PREDIÇÃO
result_predicao = modelo.predict(dados_concat)
print("\n--- RESULTADO DA PREDIÇÃO ---")
print(f"Classe Predita: {result_predicao[0]}")
print("\nClasses possíveis do modelo:")
print(modelo.classes_)
print("\nProbabilidades por classe:")
print(modelo.predict_proba(dados_concat))