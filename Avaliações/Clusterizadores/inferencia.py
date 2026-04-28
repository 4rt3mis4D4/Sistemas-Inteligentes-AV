# Imports
import pandas as pd
import pickle
import numpy as np

# 1. Entrada de dados (recebe valores do novo dado)

# -- Salvando nomes das colunas
columns_name = ['Gender', 'Age', 'Height', 'Weight', 
                'family_history_with_overweight',
                'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 
                'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 
                'MTRANS', 'NObeyesdad']

# -- Declaração do novo dado
novo_dado = [['Female', 19, 1.65, 58, 'no', 'no', 3, 3, 
                'Frequently', 'no', 3, 'no', 3, 0, 'no', 
                'Public_Transportation', 'Normal_Weight']]

# -- Cria um dataframe com os dados novos e estrutura desejada
dados_dataframe = pd.DataFrame(novo_dado, columns=columns_name)

# 2. Pré-Processamento (aplica a normalização salva anteriormente)

# -- Carregamento dos arquivos (normalizador e modelo treinado)
normalizador = pickle.load(open('normalizador.pkl', 'rb'))
normalizador_cat_ord = pickle.load(open('normalizador_cat_ord.pkl', 'rb'))
normalizador_car_nom = pickle.load(open('normalizador_cat_nom.pkl', 'rb'))
cluster = pickle.load(open('cluster.pkl', 'rb'))

# -- Separa atributos numéricos e atributos categóricos
# Numéricos: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE 
dados_num = dados_dataframe.drop(columns=['Gender', 
                                'family_history_with_overweight', 
                                'FAVC', 
                                'CAEC', 
                                'SMOKE', 
                                'SCC', 
                                'CALC', 
                                'MTRANS',
                                'NObeyesdad'])
# Categóricos Ordinais: CAEC, CALC, NObeyesdad
dados_cat_ord = dados_dataframe[['CAEC', 'CALC', 'NObeyesdad']]

# Categóricos Nominais: Gender, family_history, FAVC, SMOKE, SCC, MTRANS
dados_cat_nom = dados_dataframe [['Gender', 
                       'family_history_with_overweight', 
                       'FAVC', 
                       'SMOKE', 
                       'SCC', 
                       'MTRANS']]

# -- Normalizar novos dados numéricos
novos_dados_num_norm = normalizador.transform(dados_num)

# -- Normalizar dados categóricos ordinais
dados_cat_ord_norm = dados_cat_ord.copy()

for col in dados_cat_ord.columns:
    encoder_da_coluna = normalizador_cat_ord[col]
    
    dados_cat_ord_norm[col] = encoder_da_coluna.transform(dados_cat_ord[col])
    
# -- Normalizar dados categóricos nominais 
dados_cat_nom_norm = pd.get_dummies(dados_cat_nom, prefix_sep='_', dtype=int)

# -- Recuperamos as colunas que o modelo espera (as que salvamos no treino)
colunas_treino = normalizador_car_nom.columns

# -- Converte novos dados numéricos para dataframe
novos_dados_num_norm = pd.DataFrame(novos_dados_num_norm, columns=dados_num.columns)

# -- Garante que o novo dado tenha as mesmas colunas do One-Hot do treino
dados_cat_nom_norm = dados_cat_nom_norm.reindex(columns=normalizador_car_nom.columns, fill_value=0)

# -- Junta tudo garantindo a ordem correta
novos_dados_norm = pd.concat([novos_dados_num_norm, dados_cat_nom_norm, dados_cat_ord_norm], axis=1)

# 3. Predição (recebe os novos dados e identifica qual cluster possui proximidade)
cluster_dados_novos = cluster.predict(novos_dados_norm)
print(cluster_dados_novos)
