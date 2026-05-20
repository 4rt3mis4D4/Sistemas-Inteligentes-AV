#Classificador - versão 1
#Arquivo de dados: fertility_diagnosys.txt
#Versão sem balanceamento de classes

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn .metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pickle import load, dump
#Abrir o arquivo de dados
dados = pd.read_csv('fertility_Diagnosis.txt', sep = ',')
#Separar atributos e classe
dados_atributos = dados.drop(columns=['Diagnostico'])
dados_classe = dados['Diagnostico']

#segmentar os dados em dados para treinamento e dados para teste
atributos_train, atributos_teste, classe_train,classe_test = train_test_split(dados_atributos,dados_classe, test_size=0.3)

#=======================
#TREINAR O MODELO
tree = DecisionTreeClassifier(random_state=42)
fertility_tree = tree.fit(atributos_train, classe_train)

#Salvar o modelo
dump(fertility_tree, 
     open('fertilty_tree.pkl', 'wb'))

#Testando o modelo
diagnostico_predito = \
    fertility_tree.predict(atributos_teste)
 
#Acurácia geral
acuracia = accuracy_score(classe_test, diagnostico_predito)
print('acurácia:', acuracia)

#Matriz de contingência
ConfusionMatrixDisplay.from_estimator(fertility_tree,atributos_teste, classe_test)
plt.show()

#Calcular especificidade e sensibilidade
tn, fp, fn, tp = confusion_matrix(classe_test, diagnostico_predito).ravel()
#especificidade = vn/(vn+fp)
especificidade = tn/(tn+fp)

#sensibilidade = vp/(vp+fn)
sensibilidade = tp/(tp+fn)

print('especificidade: ', especificidade)
print('sensibiliade:', sensibilidade)
