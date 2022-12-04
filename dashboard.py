import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import poisson

st.set_page_config(layout="centered", 
                    page_title="Previsão Copa do Mundo - Catar 2022")

st.title("Previsão Copa do Mundo - Catar 2022")
st.text("")
st.text("")

selecoes = pd.read_excel('DadosCopaDoMundoQatar2022.xlsx', sheet_name='selecoes', index_col=0)
jogos = pd.read_excel('DadosCopaDoMundoQatar2022.xlsx', sheet_name='jogos', index_col=0)
vencer = pd.read_excel('outputSimulacoesCopadoMundo.xlsx', sheet_name='Sheet1', index_col=0)
etapas = pd.read_excel('outputProbabilidadesPorEtapa.xlsx', sheet_name='Sheet1', index_col=0)
avanco = pd.read_excel('outputAvancoPorEtapa.xlsx', sheet_name='Sheet1', index_col=0)

times1 = selecoes.index.to_list()
times2 = times1.copy()

# Tratando os dados
elorating = np.array(selecoes['PontosEloRating']).reshape(-1, 1)
MinMax = MinMaxScaler(feature_range=(0.15, 1))
newelorating = MinMax.fit_transform(elorating)

fifarating = np.array(selecoes['PontosRankingFIFA']).reshape(-1, 1)
newfifarating = MinMax.fit_transform(fifarating)

ref = (newelorating + newfifarating) / 2

selecoes["forca"] = ref

forca = selecoes["forca"]

# Encontrando M1 e M2
def MediasPoisson(selecao1, selecao2, data=forca, mgols=2.5):
    # Pegando do Dataframe a força de cada seleção
    forca1 = forca[selecao1]
    forca2 = forca[selecao2]
    
    # Obtenção dos gols para o time 1
    l1 = mgols*forca1/(forca1+forca2)
    # Obtenção dos gols para o time 2
    l2 = mgols - l1
    # Retorno
    return l1, l2

def Jogo(selecao1, selecao2):
    # Pegando as médias de gols para cada seleção da função anterior
    l1, l2 = MediasPoisson(selecao1, selecao2)
    
    # Obtendo de uma Poisson um valor inteiro randômico para gols do time 1, a partir da sua média
    gols1 = int(np.random.poisson(lam = l1, size=1))
    # Obtendo de uma Poisson um valor inteiro randômico para gols do time 2, a partir da sua média
    gols2 = int(np.random.poisson(lam = l2, size=1))
    
    # Saldo de gols do time 1
    saldo1 = gols1 - gols2
    # Salo de gols do time 2
    saldo2 = gols2 - gols1
    
    # Obtenção da quantidade de Pontos para cada seleção
    pontos1, pontos2, resultado = Pontos(gols1, gols2)
    
    # Formato de exibição
    placar = '{}x{}'.format(gols1, gols2)
    
    # Retorno dos: gols, saldo, pontos e o placar
    return [gols1, gols2, saldo1, saldo2, pontos1, pontos2, resultado, placar]

# FUNÇÕES AUXILIARES
def Resultado(gols1, gols2):
    # Se o time 1 fez mais gols do que o time 2, então ele V = VENCEU
    if gols1 > gols2:
        resultado = 'V'
    # Se o time 1 fez a mesma quantidade de gols do que o time 2, então ele E = Empatou
    elif gols1 == gols2:
        resultado = 'E'
    # Se não aconteceu nenhum dos dois, ele perdeu
    else:
        resultado = 'D'
    return resultado

def Pontos(gols1, gols2):
    rst = Resultado(gols1, gols2)
    # Se a saída da Função Resultado for vitória, o time 1 ganhou 3 pontos e o time 2 não ganhou nada
    if rst == 'V':
        pontos1, pontos2 = 3, 0
    # Se a saída da Função Resultado for empate, o time 1 ganhou 1 pontos e o time 2 ganhou 1
    if rst == 'E':
        pontos1, pontos2 = 1, 1
    # Se a saída da Função Resultado for derrota, o time 1 ganhou 0 pontos e o time 2 ganhou 3
    if rst == 'D':
        pontos1, pontos2 = 0, 3
    return [pontos1, pontos2, rst]

def Distribuicao(media):
    # Lista para colocarmos as probabilidade
    probs = []
    
    # Executar a função de Poisson para a média de gols do time até x = 7.
    for i in range(7):
        probs.append(poisson.pmf(i, media))
    # Obtendo a probabilidade da quantidade de gols ser maior ou igual a 7
    probs.append(1 - sum(probs))
    
    # Retorno das probabilidades
    return pd.Series(probs, index=['0', '1', '2', '3', '4', '5', '6', '7+'])

def ProbabilidadesPartidas(selecao1, selecao2):
    
    # Gerando a média para cada seleção
    l1, l2 = MediasPoisson(selecao1, selecao2)
    # Usando a função anterior para gerar a distribuição de probabilidade
    d1, d2 = Distribuicao(l1), Distribuicao(l2)
    # Gerando uma matriz da multiplicação das probabilidades
    matriz = np.outer(d1, d2)
    # Somando o triangulo inferior para a probabilidade de vitória do time 1
    vitoria = np.tril(matriz).sum() - np.trace(matriz)  #Soma o triâgulo inferior
    # Somando o triângulo superior para a probabilidade de derrota do time 1
    derrota = np.triu(matriz).sum() - np.trace(matriz)  #Soma o triângulo superior
    # Obtedo a probabilidade de empate
    empate = 1 - (vitoria + derrota)
    
    # Arredondando para 3 casas decimais e definindo o padrão de porcentagem
    probs = np.around([vitoria, empate, derrota], 3)
    probsp = [f'{100*i:.1f}%' for i in probs]
    
    # Transformando a matriz de multiplicação em um dataframe e modificando os nomes dos indices e das colunas
    nomes = ['0', '1', '2', '3', '4', '5', '6', '7+']
    matriz = pd.DataFrame(matriz, columns=nomes, index=nomes)
    matriz.index = pd.MultiIndex.from_product([[selecao1], matriz.index])
    matriz.columns = pd.MultiIndex.from_product([[selecao2], matriz.columns])
    
    # Saída da função um dicionário
    output = {
        'seleção1': selecao1,
        'seleção2': selecao2,
        'f1': forca[selecao1],
        'f2': forca[selecao2],
        'media1': l1,
        'media2': l2,
        'probabilidades': probsp,
        'matriz': matriz
    }
    return output

def GerarHeatmap(selecao1, selecao2):
    tab = ProbabilidadesPartidas(selecao1, selecao2)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax = sbn.heatmap(tab['matriz'], fmt=".5f", annot=True, cmap="crest", ax=ax,
                    yticklabels=[0, 1, 2, 3, 4, 5, 6, "7+"], xticklabels=[0, 1, 2, 3, 4, 5, 6, "7+"])

    ax.set_xlabel(tab["seleção2"], fontsize=16, labelpad=15, loc='left')
    ax.set_ylabel(tab["seleção1"], fontsize=16, labelpad=15, loc='top')
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")

    st.pyplot(fig)

## Aplicativo

# Definindo as seleções
coluna_esquerda_1, coluna_esquerda_2, coluna_direita_1, coluna_direita_2  = st.columns(4)

with coluna_esquerda_1:
    selecao1 = st.selectbox("Escolha uma Seleção:",times1)
    st.metric("Nº de Copas", selecoes.loc[selecao1, 'Copas'])
    
#st.metric("% Vencer a Copa", vencer.loc[selecao1, 'Campeão'].round(2) * 100)
#st.metric("% Chegar a Final", vencer.loc[selecao1, 'Final'].round(2) * 100)

with coluna_esquerda_2:
    st.image(selecoes.loc[selecao1, 'LinkBandeiraGrande'], width=150)

#st.metric("% Chegar a Final", vencer.loc[selecao1, 'Final'].round(2) * 100)

times2.remove(selecao1)

with coluna_direita_2:
    selecao2 = st.selectbox("Escolha uma Seleção:",times2)
    st.metric("Nº de Copas", selecoes.loc[selecao2, 'Copas'])
    

with coluna_direita_1:
    st.image(selecoes.loc[selecao2, 'LinkBandeiraGrande'], width=150)

jogoentre = ProbabilidadesPartidas(selecao1, selecao2)['probabilidades']

coluna_1, coluna_2, coluna_3 = st.columns(3)

coluna_1.markdown('<h3 align="center">'+selecao1+'</h3>', unsafe_allow_html=True)
coluna_1.markdown('<p align="center" style="font-size:35px">'+jogoentre[0]+'</p>', unsafe_allow_html=True)
coluna_1.text(" ")
coluna_1.text(" ")
coluna_1.text(" ")
coluna_1.text(" ")
coluna_1.text(" ")
coluna_1.markdown('<h3 align="center">Chegar à Final</h3>', unsafe_allow_html=True)
coluna_1.markdown('<p align="center" style="font-size:35px">'+"%.2f" % (vencer.loc[selecao1, "Final"]*100)+'%</p>', unsafe_allow_html=True)
coluna_1.markdown('<h3 align="center">Levantar a Taça</h3>', unsafe_allow_html=True)
coluna_1.markdown('<p align="center" style="font-size:35px">'+"%.2f" % (vencer.loc[selecao1, "Campeão"]*100)+'%</p>', unsafe_allow_html=True)

coluna_2.markdown('<h3 align="center">Empate</h3>', unsafe_allow_html=True)
coluna_2.markdown('<p align="center" style="font-size:35px">'+jogoentre[1]+'</p>', unsafe_allow_html=True)
coluna_2.markdown('<h3 align="center">% na Copa</h3>', unsafe_allow_html=True)

coluna_3.markdown('<h3 align="center">'+selecao2+'</h3>', unsafe_allow_html=True)
coluna_3.markdown('<p align="center" style="font-size:35px">'+jogoentre[2]+'</p>', unsafe_allow_html=True)
coluna_3.text(" ")
coluna_3.text(" ")
coluna_3.text(" ")
coluna_3.text(" ")
coluna_3.text(" ")
coluna_3.markdown('<h3 align="center">Chegar à Final</h3>', unsafe_allow_html=True)
coluna_3.markdown('<p align="center" style="font-size:35px">'+"%.2f" % (vencer.loc[selecao2, "Final"]*100)+'%</p>', unsafe_allow_html=True)
coluna_3.markdown('<h3 align="center">Levantar a Taça</h3>', unsafe_allow_html=True)
coluna_3.markdown('<p align="center" style="font-size:35px">'+"%.2f" % (vencer.loc[selecao2, "Campeão"]*100)+'%</p>', unsafe_allow_html=True)

coluna_11, coluna_12, colunameio, coluna_21, coluna_22 = st.columns(5)
## Métricas
with coluna_11:
    st.subheader('Ranking Fifa')
    st.metric("Posição", selecoes.loc[selecao1, "PosiçãoRankingFIFA"])

    st.markdown("---")
    st.metric("Posição", selecoes.loc[selecao1, "PosiçãoEloRating"])

    st.subheader("Destaque")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.subheader(selecoes.loc[selecao1, "JogadorDestaque"])

with coluna_12:
    st.markdown("---")
    st.metric("Pontos", selecoes.loc[selecao1, "PontosRankingFIFA"])

    st.subheader('Raking Elo')
    st.metric("Pontos", selecoes.loc[selecao1, "PontosEloRating"])

    st.markdown("---")
    st.image(selecoes.loc[selecao1, 'FotoJogadorDestaque'], width=150)

with coluna_21:
    st.subheader('Ranking Fifa')
    st.metric("Posição", selecoes.loc[selecao2, "PosiçãoRankingFIFA"])

    st.markdown("---")
    st.metric("Posição", selecoes.loc[selecao2, "PosiçãoEloRating"])

    st.subheader("Destaque")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.subheader(selecoes.loc[selecao2, "JogadorDestaque"])

with coluna_22:
    st.markdown("---")
    st.metric("Pontos", selecoes.loc[selecao2, "PontosRankingFIFA"])

    st.subheader('Raking Elo')
    st.metric("Pontos", selecoes.loc[selecao2, "PontosEloRating"])

    st.markdown("---")
    st.image(selecoes.loc[selecao2, 'FotoJogadorDestaque'], width=150)

st.header("Probabilidade dos Placares")
GerarHeatmap(selecao1=selecao1, selecao2=selecao2)
