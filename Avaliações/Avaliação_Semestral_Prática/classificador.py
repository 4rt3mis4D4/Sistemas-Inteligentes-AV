# IMPORTS
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate, StratifiedKFold
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. CARREGAR DATASET
# -- Abrir arquivo
dados = pd.read_csv('mushroom.csv', sep=";")
# -- Tratamento da coluna com dados incompletos
dados['stalk-root'] = dados['stalk-root'].replace('?', 'u')

# 2. REMOÇÃO DE VARIÁVEIS QUE PODEM CAUSAR RUÍDO
dados_remov = ['veil-type']
dados = dados.drop(columns=dados_remov)

# 3. SEPARAR DADOS CATEGÓRICOS + DEFINIR VARIÁVEL ALVO
# -- Categóricos Ordinais
dados_cat_ord = ['ring-number', 'gill-spacing', 'gill-size']
# -- Categóricos Nominais
dados_cat_nom = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-color', 'stalk-shape', 'stalk-root', 
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 
    'ring-type', 'spore-print-color', 'population', 'habitat'
]
# -- Definindo variável alvo 
dado_alvo = 'mushroom_type'

# 4. SEPARAR ATRIBUTOS (X) E TARGET (y)
X = dados.drop(columns=[dado_alvo])
y = dados[dado_alvo]

# 5. NORMALIZAÇÃO
# -- Categórica Ordinal
dict_encoders = {}
X_ord = X[dados_cat_ord].copy()

for col in dados_cat_ord:
    encoder = LabelEncoder()
    X_ord[col] = encoder.fit_transform(X[col])
    dict_encoders[col] = encoder
pickle.dump(dict_encoders, open('normalizador_cat_ord.pkl', 'wb'))

# -- Categórica Nominal
X_nom = pd.get_dummies(X[dados_cat_nom], prefix=dados_cat_nom, dtype=int)
pickle.dump(X_nom.columns.to_list(), open('normalizador_cat_nom.pkl', 'wb'))

# -- Concatenação 
X_final = pd.concat([X_ord, X_nom], axis=1)

# 6. BALANCEAMENTO
balancer = SMOTE(random_state=42)
X_resampled, y_resampled = balancer.fit_resample(X_final, y)
print("\nDistribuição após SMOTE:", Counter(y_resampled))

# 7. HIPERPARAMETRIZAÇÃO COM RANDOM FOREST
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    'criterion': ['gini', 'entropy'],
    'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    'max_features': ['sqrt', 'log2']
}
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
rf_hyperparameters.fit(X_resampled, y_resampled)
print("\nMelhores Parâmetros:")
pprint(rf_hyperparameters.best_params_)

# 8. DEFINIR O MODELO COM OS MELHORES PARÂMETROS
best_rf = rf_hyperparameters.best_estimator_

# 9. CROSS-VALIDATION
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
scores_cross = cross_validate(
                            best_rf,
                            X_resampled, 
                            y_resampled,
                            scoring=scoring,
                            n_jobs=-1,
                            cv=10,
                            verbose=1
)
print('\nResultado do cross validation:', scores_cross)
print('\nAcurácia CV:', scores_cross['test_accuracy'].mean())
print('Precision CV:', scores_cross['test_precision_macro'].mean())
print('Recall CV:', scores_cross['test_recall_macro'].mean())
print('F1 Score CV:', scores_cross['test_f1_macro'].mean())

# 10. TREINANDO O MODELO FINAL
best_rf.fit(X_resampled, y_resampled)
pickle.dump(best_rf, open('modelo_rf_mushroom.pkl', 'wb'))
print("\nModelo treinado e salvo com sucesso!")
