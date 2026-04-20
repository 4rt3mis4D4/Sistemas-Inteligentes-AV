# IMPORTS
import pandas as pd
import pickle

# 1. Dados para criar os data frames
columns_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV', 'CHAS_0.0', 'CHAS_1.0']

# 2. Cria um dataframe vazio, com a estrutura desejada
imovel_dataframe = pd.DataFrame(columns=columns_names)

# 3. Declarando novo dado para o dataset
novo_imovel = [[0.00632, 18.0, 2.31, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98, 24.0]]

# 4. Abrir o normalizador + modelo salvo
normalizador = pickle.load(open('normalizador_hd.pkl', 'rb'))
cluster_hd = pickle.load(open('cluster_hd.pkl', 'rb'))

# 5. Normalizar os dados de entrada
novo_imovel_norm = normalizador.transform(novo_imovel)
#print(novo_imovel_norm)

# 6. Converter a nova instancia normalizada em dataframe
colunas_numericas = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

novo_imovel_norm_df = pd.DataFrame(novo_imovel_norm, columns=colunas_numericas)

# 7. Adicionar a informação do CHAS ao DataFrame normalizado antes da concatenação
chas_valor = 0.0

if chas_valor == 1.0:
    novo_imovel_norm_df['CHAS_1.0'] = 1
else:
    novo_imovel_norm_df['CHAS_0.0'] = 1

# 8. Concatenar com o DataFrame estruturado e preencher o que sobrar com 0
imovel_nova_instancia = pd.concat([novo_imovel_norm_df, imovel_dataframe], sort=False).fillna(0)

imovel_nova_instancia = imovel_nova_instancia[columns_names] # --- Garantir ordem das colunas exata do modelo
#print(imovel_nova_instancia)

# 9. Predição
cluster_imovel = cluster_hd.predict(imovel_nova_instancia)
print('Cluster novo imovel: ', cluster_imovel)