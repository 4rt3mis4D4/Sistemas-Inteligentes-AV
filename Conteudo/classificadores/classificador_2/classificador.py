#Classificador - versão 2
#Arquivo de dados: fertility_diagnosys.txt
#Versão com balanceamento de classes

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn .metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from pickle import load, dump
from imblearn.over_sampling import SMOTE
import numpy as np

#Abrir o arquivo de dados
dados = pd.read_csv('fertility_Diagnosis.txt', sep = ',')
#Separar atributos e classe
dados_atributos = dados.drop(columns=['Diagnostico'])
dados_classe = dados['Diagnostico']

# Balancear os dados
balancer = SMOTE()
atributos_balanceados, classes_balanceadas = balancer.fit_resample(dados_atributos, dados_classe)
print('### FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ###')
from collections import Counter
class_count = Counter(classes_balanceadas)
class_count
dados.columns # Utilizar os rótulos das colunas

#print(classes_balanceadas.value_counts())

#segmentar os dados em dados para treinamento e dados para teste
#atributos_train, atributos_teste, classe_train,classe_test = train_test_split(dados_atributos,dados_classe, test_size=0.3)
atributos_train, atributos_teste, classe_train,classe_test = train_test_split(atributos_balanceados,classes_balanceadas, test_size=0.3)

#=======================
#TREINAR O MODELO
tree = DecisionTreeClassifier(random_state=42)
fertility_tree = tree.fit(atributos_train, classe_train)

# Hiperparametrização da Random Forest
# -- Definir os domínios para os hiperparâmetros
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

# Criar a grade de valores
rf_grid={
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier()
rf_hyperparameters = RandomizedSearchCV(
    estimator = rf,
    param_distributions= rf_grid,
    n_iter= 10,
    cv= 3,
    verbose= 2,
    n_jobs= -1
)
rf_hyperparameters.fit(dados_atributos, dados_classe)

# Mostrar o resultado da hiperparametrização 
from pprint import pprint
print('Melhores parâmetros:')
pprint(rf_hyperparameters.best_params_)
# Instanciar o estimador
rf = RandomForestClassifier(**rf_hyperparameters.best_params_)

fertility_rf = rf.fit(atributos_train, classe_train)

#Salvar o modelo
#dump(fertility_tree, 
#     open('fertilty_tree.pkl', 'wb'))

dump(fertility_rf, 
     open('fertilty_rf.pkl', 'wb'))

#Testando o modelo
#diagnostico_predito = \
#    fertility_tree.predict(atributos_teste)
    
diagnostico_predito_rf = \
    fertility_rf.predict(atributos_teste)
 
#Acurácia geral
#acuracia = accuracy_score(classe_test, diagnostico_predito)
#print('acurácia tree:', acuracia)

acuracia = accuracy_score(classe_test, diagnostico_predito_rf)
print('acurácia rf:', acuracia)
print()

#Matriz de contingência
#ConfusionMatrixDisplay.from_estimator(fertility_tree,atributos_teste, classe_test)
#plt.show()

#Calcular especificidade e sensibilidade 
#tn, fp, fn, tp = confusion_matrix(classe_test, diagnostico_predito).ravel()
#especificidade = vn/(vn+fp)
#especificidade = tn/(tn+fp)

#sensibilidade = vp/(vp+fn)
#sensibilidade = tp/(tp+fn)

#print('especificidade tree: ', especificidade)
#print('sensibiliade tree:', sensibilidade)
#print()

#Calcular especificidade e sensibilidade da random forest
tn, fp, fn, tp = confusion_matrix(classe_test, diagnostico_predito_rf).ravel()
#especificidade = vn/(vn+fp)
especificidade = tn/(tn+fp)

#sensibilidade = vp/(vp+fn)
sensibilidade = tp/(tp+fn)

print('especificidade rf: ', especificidade)
print('sensibiliade rf:', sensibilidade)
