# IMPORTS
import pandas as pd
import numpy as np 

from ucimlrepo import fetch_ucirepo

from imblearn.over_sampling import SMOTE
from collections import Counter
from pprint import pprint
from pickle import dump

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate

# ABRIR O DADOS
# -- Dataset Bank Marketing 
bank_marketing = fetch_ucirepo(id=222)
#print(bank_marketing.variables)

# CLASSIFICAR E SEPARAR OS DADOS NUMÉRICOS DOS CATEGÓRICOS
# -- Separar atributos e classe
x_raw = bank_marketing.data.features.copy()
y_raw = bank_marketing.data.targets.copy()

# -- Separação dos atributos numéricos e categórico para normalização
colunas_num = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
dados_num = x_raw[colunas_num]

colunas_ord = ['education']
dados_cat_ord = x_raw[colunas_ord]

colunas_nom = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month', 'poutcome']
dados_cat_nom = x_raw[colunas_nom]

# NORMALIZAR DADOS
# -- Normalização MinMaxScaler
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)
dump(normalizador, open('normalizador_num.pkl', 'wb'))

dados_num_norm = normalizador.transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

# -- Normalização LabelEncoder
dict_encoders = {}
dados_cat_ord_copy = dados_cat_ord.copy()

for col in dados_cat_ord.columns: 
    le = LabelEncoder()
    dados_cat_ord_copy[col] = le.fit_transform(dados_cat_ord[col].astype(str))
    dict_encoders[col] = le

dados_cat_ord_norm = dados_cat_ord_copy
dump(dict_encoders, open('normaliazdor_cat_ord.pkl', 'wb'))

# -- Normalização One-Hot Encoding 
dados_cat_nom_norm = pd.get_dummies(dados_cat_nom, prefix_sep='_', dtype=int)
dump(dados_cat_nom_norm, open('normalizador_cat_nom.pkl', 'wb'))

# UNIR DADOS JÁ NORMALIZADOS EM UM DATAFRAME
# -- Une todos os dados normalizados em um único Dataframe
dados_norm = pd.concat([dados_num_norm, dados_cat_nom_norm, dados_cat_ord_norm], axis=1)

# -- Prepara a variável alvo (y) convertendo de texto para binário (0 e 1)
dados_classe = y_raw.iloc[:, 0].map({'yes': 1, 'no': 0})

# BALANCEAMENTO
# -- Balanceamento dos dados
resampler = SMOTE(random_state=42)
atributos_b, classes_b = resampler.fit_resample(dados_norm, dados_classe)
class_count = Counter(classes_b)
print(class_count)

# HIPERPARAMETRIZAÇÃO
# -- Hiperparametrização da Random Forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

rf_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

# APLICAR RANDOM FOREST
rf = RandomForestClassifier(random_state=42)
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

rf_hyperparameters.fit(atributos_b, classes_b)

print('\nMelhores parâmetros:')
pprint(rf_hyperparameters.best_params_)

rf = RandomForestClassifier(**rf_hyperparameters.best_params_, random_state=42)

# ACURÁCIA
# -- Iniciar a avaliação cruzada da acurácia do modelo
scoring = ['accuracy', 'f1_macro', 'precision', 'recall']
scores_cross = cross_validate(
    rf,
    atributos_b,
    classes_b,
    scoring=scoring,
    n_jobs=-1,
    cv=10,
    verbose=1
)

print('\n--- Resultado do Cross Validation ---')
print('Acurácia média:', scores_cross['test_accuracy'].mean())
print('Precision média:', scores_cross['test_precision'].mean())
print('Recall médio:', scores_cross['test_recall'].mean())
print('F1 Score médio:', scores_cross['test_f1_macro'].mean())

# TREINAMENTO FINAL
# -- Treinamento pós acurácia
bank_rf = rf.fit(atributos_b, classes_b)

# -- Salvar o modelo Random Forest
dump(bank_rf, open('bank_marketing_rf.pkl', 'wb'))
print("\nModelo 'bank_marketing_rf.pkl' salvo com sucesso!")
