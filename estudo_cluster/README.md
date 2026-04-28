# 🎯 Clusterização de Clientes

Diretório para prática de **Clustering** com datasets do Kaggle.

---

## 📂 Scripts

### 🧩 `cluster.py`

Treinamento do modelo:

* Leitura do `.csv`
* Normalização (MinMaxScaler)
* Elbow Method (K ideal)
* Treino do K-Means
* Salvamento (`.pkl` com Pickle)

---

### 🔍 `descritor_cluster.py`

Interpretação dos clusters:

* Carrega modelo e scaler
* Desnormaliza dados
* Reconverte categorias
* Exibe médias por cluster

---

### 🚀 `inferencia.py`

Uso do modelo:

* Recebe novo dado
* Aplica pré-processamento
* Prediz cluster
* Retorna resultado

---

## 📦 `requirements.txt`

Dependências do projeto:

* Lista bibliotecas e versões
* Permite recriar o ambiente

Instalação:

```bash
pip install -r requirements.txt
```

---

## 📊 `arquivo.csv`

Dataset do projeto:

* Base usada no treinamento
* Contém dados dos clientes (numéricos/categóricos)
* Utilizado para gerar os clusters

---
