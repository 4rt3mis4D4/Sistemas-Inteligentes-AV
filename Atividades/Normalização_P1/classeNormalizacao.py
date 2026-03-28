# Sistemas Inteligentes Avamçados - Classe Normalização (MinMaxScaler, Label Encoder e OneHot Encoder)
# Gabriela Pedroso 20/03 - Ciência da Computação

# --- Bibliotecas
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class NormalizarDados:
    def __init__(self):
        self.scaler = MinMaxScaler() # Instancia o normalizador MinMaxScaler
        self.encoder = LabelEncoder()
        self.colunas_one_hot = []
    
    def normalizar(self, df, treinar=True):
        df_resultado = df.copy() # Copia de segurança
        colunas_numericas = ['idade', 'altura', 'peso'] # Definindo colunas numéricas
        
        # Verifica se precisa ser treinado ou apenas normalizado
        if treinar: 
            # 1. MinMaxScaler (colunas numéricas)
                # --- Treina o modelo normalizador para uso posterior
            scaler_model = self.scaler.fit(df_resultado[colunas_numericas])
                # --- Normalizar os dados
            df_resultado[colunas_numericas] = self.scaler.fit_transform(df_resultado[colunas_numericas])

            # 2. Label Encoder (colunas categóricas)
                # --- Normaliza os dados: Aprende e aplica o mapeamento
            #df_resultado['sexo'] = self.encoder.fit_transform(df_resultado['sexo'])

            # 3. One Hot Encondig 
                # --- Normaliza os dados: Transforma cada categoria em uma coluna binária
            df_resultado = pd.get_dummies(df_resultado, columns=['sexo'], prefix='sexo', prefix_sep='_', dtype=int)
                # --- Percorre a lista uma por uma das colunas e salva apenas colunas categóricas
            self.colunas_one_hot = [c for c in df_resultado.columns if c.startswith('sexo_')]
        else:
            df_resultado[colunas_numericas] = self.scaler.transform(df_resultado[colunas_numericas]) # MinMaxScaler
            #df_resultado['sexo'] = self.encoder.transform(df_resultado['sexo']) # Label Encoder

            # One Hot Enconding
            df_resultado = pd.get_dummies(df_resultado, columns=['sexo'], prefix='sexo', prefix_sep='_', dtype=int)

                # --- Garante que todas as colunas do treino existam (preenche com 0 se faltar)
            for col in self.colunas_one_hot:
                if col not in df_resultado.columns:
                    df_resultado[col] = 0
                # --- Garante a ordem exata das colunas para não confundir o modelo
            colunas_fixas = [c for c in df_resultado.columns if c not in self.colunas_one_hot] + self.colunas_one_hot
            df_resultado = df_resultado[colunas_fixas]

        return df_resultado

    def salvar(self, nome_arquivo):
        with open(nome_arquivo, 'wb') as arquivo:
            pickle.dump(self, arquivo)
        print(f"Modelo guardado com sucesso em: {nome_arquivo}")
        print()
    
    @staticmethod
    def carregar(nome_arquivo):
        with open(nome_arquivo, 'rb') as arquivo:
            return pickle.load(arquivo)
    
    def desnormalizar(self, df_norm):
        df_revertido = df_norm.copy()

        # Desnormalizando MinMaxScaler
        colunas_numericas = ['idade', 'altura', 'peso']
        df_revertido[colunas_numericas] = self.scaler.inverse_transform(df_revertido[colunas_numericas])
    
        # Desnormalizando Label Encoder
        #df_revertido['sexo'] = self.encoder.inverse_transform(df_revertido['sexo'])
        
        # Desnormalizando One Hot
            # --- Isola apenas as colunas dummies
        df_colunas_one_hot = df_revertido[self.colunas_one_hot]
            #--- Reverte para uma única coluna (Sexo)
        df_categoria_revertida = pd.from_dummies(df_colunas_one_hot, sep='_')

            # --- Adiciona a coluna original de volta
        df_revertido['sexo'] = df_categoria_revertida
            # --- Remove colunas binárias
        df_revertido = df_revertido.drop(columns=self.colunas_one_hot)

        return df_revertido

if __name__ == "__main__":
    arquivo = 'dados_normalizar.csv'

    # --- Leitura do arquivo
    df_original = pd.read_csv(arquivo, sep=';', decimal = ',')

    print("Arquivo Original:")
    print(df_original.head(5))
    print()

    normalizador = NormalizarDados() # Instanciando a classe
    df_treinado_norm = normalizador.normalizar(df_original) # Executando normalizar 

    # --- Normalizando 
    normalizador.salvar('normalizador_treinado.pkl')

    print("Arquivo Normalizado:")
    print(df_treinado_norm.head(5))
    print()

    # --- Desnormalizando
    df_revertido = normalizador.desnormalizar(df_treinado_norm)

    print("Arquivo Desnormalizado:")
    print(df_revertido.head(5))
    print()

    # NOVOS DADOS:
    normalizador_carregado = NormalizarDados.carregar('normalizador_treinado.pkl') # Carregando modelo ja normalizado

    # --- Declarando novos dados
    novos_dados = pd.DataFrame({
        'idade': [30],
        'altura': ['1.75'],
        'peso': [80],
        'sexo': ['M']
    })

    df_novos_norm = normalizador_carregado.normalizar(novos_dados, treinar=False)
    print("Novos Dados Normalizados:")
    print(df_novos_norm)
    print()

    df_novos_desnorm = normalizador_carregado.desnormalizar(df_novos_norm)
    print("Novos Dados Desnormalizados:")
    print(df_novos_desnorm)
    print()