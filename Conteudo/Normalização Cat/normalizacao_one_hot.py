import pandas as pd

# Criar dados para entrada
df= pd.DataFrame({
    'cor':['Vermelho', 'Azul', 'Verde', 'Azul']
})
#print(df)

# Codificação One Hot: transforma cada categoria em uma coluna binária
df_normalizado = pd.get_dummies(df, prefix='cor', prefix_sep='_', dtype=int)
print(df_normalizado)

# Exemplo: isolar uma linha do dataframe
nova_instacia = df_normalizado.iloc[1] 
# Nova instância é do tipo Pandas.Series

# Converte a Pandas.Series em DataFrame
df_nova_instancia = nova_instacia.to_frame().T
print()
print(df_nova_instancia)

# Desnormalizando
df_nova_instancia_desnormalizada = pd.from_dummies(df_nova_instancia, sep='_')
print()
print(df_nova_instancia_desnormalizada)
