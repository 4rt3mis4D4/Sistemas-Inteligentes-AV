# Imports
import pickle
import pandas as pd

# 1. Carregamento (abre arquivos: normalizador e modelo treinado)

cluster_model = pickle.load(open('cluster.pkl', 'rb')) # -- Abre modelo treinado
normalizador = pickle.load(open('normalizador.pkl', 'rb')) # -- Abre o normalizador

# 2. Desnormalização (converte os centroides de volta para a escala original)

columns_name = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
       'Gender_Female', 'Gender_Male', 'family_history_with_overweight_no',    
       'family_history_with_overweight_yes', 'FAVC_no', 'FAVC_yes', 'SMOKE_no',
       'SMOKE_yes', 'SCC_no', 'SCC_yes', 'MTRANS_Automobile', 'MTRANS_Bike',   
       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking',   
       'CAEC', 'CALC', 'NObeyesdad']

# -- Converte os centroides em dataframe
dataframe = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_name)
#print(dataframe) # -- Exibindo dataframe de centroides
# -- Desnormalizar dados numéricos normalizados anteriormente
atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(
        dataframe[
        ['Age', 
         'Height', 
         'Weight', 
         'FCVC', 
         'NCP', 
         'CH2O', 
         'FAF', 
         'TUE']]),
        columns=['Age', 
         'Height', 
         'Weight', 
         'FCVC', 
         'NCP', 
         'CH2O', 
         'FAF', 
         'TUE'])

print(atributos_num_desnorm.head(5)) # -- Exibindo dados desnormalizados

# 3. Tradução (converte as colunas categóricas de volta para o seu nome original)

# -- Selecionamos apenas as colunas Nominais (Dummies) presentes no dataframe de centroides
# Listamos as colunas que foram geradas pelo One-Hot Encoding
colunas_nominais_dummies = [
    'Gender_Female', 'Gender_Male', 
    'family_history_with_overweight_no', 'family_history_with_overweight_yes', 
    'FAVC_no', 'FAVC_yes', 
    'SMOKE_no', 'SMOKE_yes', 
    'SCC_no', 'SCC_yes', 
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 
    'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

# -- Limpeza das categorias Nominais
nominais_dummies = dataframe[colunas_nominais_dummies].round(0).astype(int)

# -- Desnormalizando dados categóricos nominais
dados_nom_desnorm = pd.from_dummies(nominais_dummies, sep='_')

# -- Tradução dos Categóricos Ordinais
dados_ord_traduzidos = dataframe[['CAEC', 'CALC', 'NObeyesdad']].round(0).astype(int)

# 4. Análise (exibe as caracteristicas médias de cada cluster criado)

# -- Junta as colunas desnormalizadas categóricas com as numéricas
clusters = pd.concat([atributos_num_desnorm, dados_nom_desnorm, dados_ord_traduzidos], axis=1)
print(clusters) # -- Exibe as caracteristicas médias de cada cluster criado
