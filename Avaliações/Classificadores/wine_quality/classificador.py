# IMPORTS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump 

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from pprint import pprint

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. CARREGAR DADOS
# -- Carregar os dois datasets
dados_vinho_tinto = pd.read_csv('datasets/winequality-red.csv', sep=';')
dados_vinho_branco = pd.read_csv('datasets/winequality-white.csv', sep=';')

# -- Unificando datasets
dados = pd.concat([dados_vinho_tinto, dados_vinho_branco], ignore_index=True)
#print(dados.head(5))

# -- Separar atributos (X) e classes (Y)
dados_atributos = dados.drop(columns=['quality'])
dados_classe = dados['quality']

# 2. NORMALIZAÇÃO
# -- Instanciando normalizador
scaler = MinMaxScaler()
# -- Treinando normalizador
normalizador = scaler.fit(dados_atributos)
# -- Salvando normalizador para uso posterior
dump(normalizador, open('normalizador.pkl', 'wb'))
# -- Normalizando os dados numéricos
dados_atributos_norm = normalizador.transform(dados_atributos)
# -- Convertendo matriz numérica em um Dataframe
dados_atributos_norm = pd.DataFrame(dados_atributos_norm, columns=dados_atributos.columns)
#print(dados_atributos_norm.head(5))

# 3. BALANCEAMENTO
# -- Construção do balanceador 
resampler = SMOTE(random_state=42, k_neighbors=4)
atributos_b, classes_b = resampler.fit_resample(dados_atributos_norm, dados_classe)

print("### FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ###")
for classe, count in sorted(Counter(dados_classe).items()):
    print(f'  Classe {classe}: {count} amostras')

# 4. SEGMENTAR OS DADOS EM DADOS PARA TREINAMENTO E DADOS PARA TESTE
atributos_train, atributos_teste, classe_train, classe_teste = train_test_split(
    atributos_b, classes_b, test_size=0.3, random_state=42
)

# 5. HIPERPARAMETRIZAÇÃO (3 ALGORITMOS)
# 5.1 Random Forest
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=2)],
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
rf_hyperparameters.fit(atributos_b, classes_b)

# -- Exibir resultado da hiperparametrização Random Forest
print('\nMelhores parâmetros (Random Forest):')
pprint(rf_hyperparameters.best_params_)
print(f'Melhor score (cv): {rf_hyperparameters.best_score_:.4f}')
# -- Instanciar o estimador Random Forest
rf_best = RandomForestClassifier(**rf_hyperparameters.best_params_, random_state=42)

# 5.2 Extra Trees
et_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=2)],
    'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    'max_features': ['sqrt', 'log2']
}

et = ExtraTreesClassifier(random_state=42)
et_hyperparameters = RandomizedSearchCV(
    estimator=et,
    param_distributions=et_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
et_hyperparameters.fit(atributos_b, classes_b)

# -- Exibir resultado da hiperparametrização Extra Trees
print('\nMelhores parâmetros (Extra Trees):')
pprint(et_hyperparameters.best_params_)
print(f'Melhor score (cv): {et_hyperparameters.best_score_:.4f}')
# -- Instanciar o estimador Extra Trees
et_best = ExtraTreesClassifier(**et_hyperparameters.best_params_, random_state=42)

# 5.3 Gradient Boosting
gb_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=50, stop=200, num=5)],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

gb = GradientBoostingClassifier(random_state=42)
gb_hyperparameters = RandomizedSearchCV(
    estimator=gb,
    param_distributions=gb_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
gb_hyperparameters.fit(atributos_b, classes_b)

# -- Exibir resultado da hiperparametrização Gradient Boosting
print('\nMelhores parâmetros (Gradient Boosting):')
pprint(gb_hyperparameters.best_params_)
print(f'Melhor score (cv): {gb_hyperparameters.best_score_:.4f}')
# -- Instanciar o estimador Gradient Boosting
gb_best = GradientBoostingClassifier(**gb_hyperparameters.best_params_, random_state=42)

# 6. AVALIAÇÃO CRUZADA DA ACURÁCIA DOS 3 MODELOS
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
resultados={}
modelos = {
    'Random Forest': rf_best,
    'Extra Trees': et_best,
    'Gradient Boosting': gb_best
}

for nome, modelo in modelos.items():
    scores_cross = cross_validate(
        modelo,
        atributos_b,
        classes_b,
        scoring=scoring,
        n_jobs=-1,
        cv=10,
        verbose=1
    )
    resultados[nome] = {
        'accuracy': scores_cross['test_accuracy'].mean(),
        'f1_macro': scores_cross['test_f1_macro'].mean(),
        'precision': scores_cross['test_precision_macro'].mean(),
        'recall': scores_cross['test_recall_macro'].mean()
    }
    print(f'\nResultado do cross_val - {nome}:')
    print(f'  Acurácia:  {resultados[nome]["accuracy"]:.4f}')
    print(f'  Precision: {resultados[nome]["precision"]:.4f}')
    print(f'  Recall:    {resultados[nome]["recall"]:.4f}')
    print(f'  F1 Score:  {resultados[nome]["f1_macro"]:.4f}')

# 7. SELEÇÃO E TREINAMENTO DO MELHOR MODELO
print('\n#### TABELA DE COMPARAÇÃO DOS MODELOS ####')
print(f"{'Modelo':<25} {'Accuracy':>10} {'F1-Score':>10}")
print('-' * 47)
for nome, m in resultados.items():
    print(f"{nome:<25} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f}")
 
# -- Selecionar automaticamente o melhor modelo com base na acurácia
melhor_metaestimador = max(resultados, key=lambda x: resultados[x]['accuracy'])
melhor_modelo_obj = modelos[melhor_metaestimador]
print(f'\nMelhor modelo selecionado automaticamente: {melhor_metaestimador}')

# -- Treina melhor modelo + salva
melhor_modelo_treinado = melhor_modelo_obj.fit(atributos_train, classe_train)
dump(melhor_modelo_treinado, open('modelo.pkl', 'wb'))

# 8. AVALIAÇÃO
# -- Predição do conjunto de teste
quality_predito = melhor_modelo_treinado.predict(atributos_teste)
# -- Acurácia Geral
acuracia = accuracy_score(classe_teste, quality_predito)
print(f'Acurácia: {acuracia:.4f}')
# -- F1-Score 
f1 = f1_score(classe_teste, quality_predito, average='macro')
print(f'F1-Score: {f1:.4f}')
# -- Matriz de Contingência
mc = confusion_matrix(classe_teste, quality_predito)
print(mc)

ConfusionMatrixDisplay.from_predictions(classe_teste, quality_predito)
plt.title(f'Matriz de Contingência - {melhor_metaestimador}')
plt.tight_layout()
plt.savefig('matriz_contingencia.png', dpi=100)
plt.close()

# -- Acurácia por classe
print('\n#### ACURÁCIA POR CLASSE ####')
print(f"{'Classe':<10} {'Acurácia (%)':>15}")
print('-' * 27)
for i, classe in enumerate(sorted(set(classe_teste))):
    vp = mc[i, i]
    total_classe = mc[i, :].sum()
    acc_classe = (vp / total_classe) * 100 if total_classe > 0 else 0
    print(f"{classe:<10} {acc_classe:>14.2f}%")
 
# 9. TABELA RESULTADO
print(f"\n{'Métrica':<25} {'Valor':>15}")
print('-' * 42)
print(f"{'Melhor Modelo':<25} {melhor_metaestimador:>15}")
print(f"{'Accuracy Global':<25} {acuracia:>14.4f}")
print(f"{'F1-Score (macro)':<25} {f1:>14.4f}")
