# IMPORTS
import pandas as pd
import numpy as np
from pickle import dump

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ABRIR DADOS
# -- Carregar dataset 
dados = pd.read_csv('diabetes.csv')
# -- Definindo atributos
colunas_atributos = [
    'Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age'
]
# -- Separar atributos (X) e a variável alvo (Y)
x = dados[colunas_atributos]
y = dados['Outcome']
# -- Divisão treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

# NORMALIZÇÃO
# -- Instanciando normalizador
scaler = MinMaxScaler()
# -- Treinamento
x_train_norm = scaler.fit_transform(x_train)
# -- Normalização
x_test_norm = scaler.transform(x_test)
# -- Salvando modelo normalizado
dump(scaler, open('scaler_diabetes.pkl', 'wb'))

# BALANCEAMENTO
smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train_norm, y_train)

# HIPERPARÂMETRIZAÇÃO
resultados = []

# -- Random Forest
rf_param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

# -- Support Vector Machine (SVM)
svm_param = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svm_search = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_param, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

# -- XGBoost
xgb_param = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
xgb_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_param, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

# -- Lista com as buscas de hiperparâmetros configurados
modelos_para_treinar = [
    ('Random Forest', rf_search),
    ('SVM', svm_search),
    ('XGBoost', xgb_search)
]

# TREINAMENTO FINAL
print("Iniciando a busca de hiperparâmetros e treino dos modelos...\n")

for nome, busca_modelo in modelos_para_treinar:
    busca_modelo.fit(x_train_bal, y_train_bal)
    melhor_modelo = busca_modelo.best_estimator_
    y_pred = melhor_modelo.predict(x_test_norm)

    # -- Cálculo das métricas de validação
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    resultados.append({
        'Modelo': nome,
        'Acurácia': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

    nome_arquivo = f"modelo_{nome.replace(' ', '_').lower()}.pkl"
    dump(melhor_modelo, open(nome_arquivo, 'wb'))

# DATAFRAME
df_resultados = pd.DataFrame(resultados).sort_values(by='Acurácia', ascending=False).reset_index(drop=True)
print("\n--- TABELA ACURÁCIA DOS 3 MODELOS ---\n")
print(df_resultados.to_string(index=False))

# Após analisar a tabela compartiva dos 3 modelos, cheguei a conclusão que o modelo "Random Forest" é o melhor, 
# pois ele obteve os melhores resultados em todas as métricas: maior Acurácia (0,7597), Recall (0,7407) e F1-Score (0,6838), 
# sendo a escolha mais segura para identificar pacientes diabéticos e minimizar falsos negativos.
