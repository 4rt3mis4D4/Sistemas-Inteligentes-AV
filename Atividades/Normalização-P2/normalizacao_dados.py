# Sistemas Inteligentes Avançados - Novos Dados + Normalização One-Hot Encoder
# Gabriela Pedroso 17/03 - Ciência da Computação

import pandas as pd

# Dados Originais
df_original = pd.DataFrame({
    'cor': ['Vermelho', 'Azul', 'Verde', 'Azul']
})

# Codificação One-Hot nos Dados Originais
# --- pd.get_dummies: cada categoria vira uma coluna binária (0 ou 1)
dados_normalizados_referencia = pd.get_dummies(df_original, prefix='cor', prefix_sep='_', dtype=int)

# Converção dos nomes das colunas para lista
# --- .columns: retorna um índice com os nomes das colunas
# --- .tolist: converte esse índice para uma lista Python comum
colunas_esperadas = dados_normalizados_referencia.columns.tolist()

print("Colunas de referência:")
print(colunas_esperadas)
print("\n" + "="*50 + "\n") # Linha separado visual

# Método para normalizar nova instância
def normalizar_nova_instancia(dados_brutos_novos, colunas_referencia, prefixo='cor', separador='_'):
    # 1. Verifica se a entrada é do tipo pandas.Series
        # --- isinstance() verfica se um objeto é de um determinado tipo
    if isinstance(dados_brutos_novos, pd.Series):
        df_novo = dados_brutos_novos.to_frame().T
    # 1.2 Verifica se a entrada é um dicionário Python
        # --- isinstance() verfica se um objeto é de um determinado tipo
    elif isinstance(dados_brutos_novos, dict):
        df_novo = pd.DataFrame([dados_brutos_novos])
    # 1.3 Se não for nenhum dos tipos esperados, lança um erro
    else:
        # --- "raise" levanta exceções intencionalmente
        raise TypeError("Entrada deve ser um dicionário ou pandas.Series")

    # 2. Aplica Codificação One Hot nos novos dados
    df_novo_normalizado = pd.get_dummies(df_novo, prefix=prefixo, prefix_sep=separador, dtype=int)

    print("One Hot aplicado na nova instância:")
    print(df_novo_normalizado)

    # 3. Alinhamento com as colunas de referência
        # --- .reindex() realizinha DataFrame com novas colunas/indíces
        # --- columns lista de colunas que queremos no resultado
        # --- fill_value Valor para preencher colunas que nçai existiam
    df_novo_final = df_novo_normalizado.reindex(columns=colunas_referencia, fill_value=0)

    return df_novo_final


# Instância nova com "pandas.Series"
nova_instancia_ps = pd.Series({'cor': 'Azul'})
print("Entrada: cor = 'Azul'")
print(f"Tipo da entrada: {type(nova_instancia_ps)}")
print(f"Conteúdo: {nova_instancia_ps}")

    # Chamada da função de normalização + resultado
resultado_ps = normalizar_nova_instancia(nova_instancia_ps, colunas_esperadas, prefixo='cor', separador='_')
print(resultado_ps)
print("\n" + "-"*30 + "\n")


# Instância nova com "dicionário"
nova_instacia_dic = {'cor': 'Amarelo'}
print("Entrada: cor = 'Amarelo'")
print(f"Tipo de entrada: {type(nova_instacia_dic)}")
print(f"Conteúdo: {nova_instacia_dic}")

    # Chamada da função de normalização + resultado
resultado_dic = normalizar_nova_instancia(nova_instacia_dic, colunas_esperadas, prefixo='cor', separador='_')
print(resultado_dic)
