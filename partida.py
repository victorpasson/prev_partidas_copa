import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Previsão Copa 2022", layout="centered")

st.title('Minha IA que prevê Jogos da Copa!')

selecoes = pd.read_excel('DadosCopaDoMundoQatar2022.xlsx', sheet_name='selecoes', index_col=0)
jogos = pd.read_excel('DadosCopaDoMundoQatar2022.xlsx', sheet_name='jogos')

# Transformação Linear de Escala Numérica
fifa = selecoes['PontosRankingFIFA']
a, b = min(fifa), max(fifa)
fa, fb = 0.15, 1

b1 = (fb - fa)/(b-a)
b0 = fb - b*b1
forca = b0 + b1*fifa

# Encontrando M1 e M2

def MediasPoisson(selecao1, selecao2):
    forca1 = forca[selecao1]
    forca2 = forca[selecao2]
    mgols = 2.75
    
    l1 = mgols*forca1/(forca1+forca2)
    l2 = mgols - l1
    return l1, l2

def Resultado(gols1, gols2):
    if gols1 > gols2:
        resultado = 'V'
    elif gols1 == gols2:
        resultado = 'E'
    else:
        resultado = 'D'
    return resultado

def Pontos(gols1, gols2):
    rst = Resultado(gols1, gols2)
    if rst == 'V':
        pontos1, pontos2 = 3, 0
    if rst == 'E':
        pontos1, pontos2 = 1, 1
    if rst == 'D':
        pontos1, pontos2 = 0, 3
    return [pontos1, pontos2, rst]

def Jogo(selecao1, selecao2):
    l1, l2 = MediasPoisson(selecao1, selecao2)
    
    gols1 = int(np.random.poisson(lam = l1, size=1))
    gols2 = int(np.random.poisson(lam = l2, size=1))
    
    saldo1 = gols1 - gols2
    saldo2 = gols2 - gols1
    
    pontos1, pontos2, resultado = Pontos(gols1, gols2)
    
    placar = '{}x{}'.format(gols1, gols2)
    
    return [gols1, gols2, saldo1, saldo2, pontos1, pontos2, resultado, placar]

def Distribuicao(media):
    probs = []
    
    for i in range(7):
        probs.append(poisson.pmf(i, media))
    probs.append(1 - sum(probs))
    return pd.Series(probs, index=['0', '1', '2', '3', '4', '5', '6', '7+'])

def ProbabilidadesPartidas(selecao1, selecao2):
    l1, l2 = MediasPoisson(selecao1, selecao2)
    d1, d2 = Distribuicao(l1), Distribuicao(l2)
    matriz = np.outer(d1, d2)
    vitoria = np.tril(matriz).sum() - np.trace(matriz)  #Soma o triâgulo inferior
    derrota = np.triu(matriz).sum() - np.trace(matriz)  #Soma o triângulo superior
    empate = 1 - (vitoria + derrota)
    
    probs = np.around([vitoria, empate, derrota], 3)
    probsp = [f'{100*i:.1f}%' for i in probs]
    
    nomes = ['0', '1', '2', '3', '4', '5', '6', '7+']
    matriz = pd.DataFrame(matriz, columns=nomes, index=nomes)
    matriz.index = pd.MultiIndex.from_product([[selecao1], matriz.index])
    matriz.columns = pd.MultiIndex.from_product([[selecao2], matriz.columns])
    
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

# Aplicativo começa agora

listaselecoes1 = selecoes.index.tolist()
listaselecoes1.sort()
listaselecoes2 = listaselecoes1.copy()

j1, j2 = st.columns(2)
selecao1 = j1.selectbox('Escolha a primeira Seleção:', listaselecoes1)
listaselecoes2.remove(selecao1)
selecao2 = j1.selectbox('Escolha a segunda Seleção:', listaselecoes2)
st.markdown('---')

jogo = ProbabilidadesPartidas(selecao1, selecao2)
prob = jogo['probabilidades']
matriz = jogo['matriz']

col1, col2, col3, col4, col5 = st.columns(5)
col1.image(selecoes.loc[selecao1, 'LinkBandeiraGrande'])
col2.metric(selecao1, prob[0])
col2.metric('Empate', prob[1])
col2.metric(selecao2, prob[2])
col1.image(selecoes.loc[selecao2, 'LinkBandeiraGrande'])

st.markdown('---')
st.markdown('## ▶️ Probabilidades dos Placares')

def aux(x):
    return f'{str(round(100*x, 1))}%'

st.table(matriz.applymap(aux))

st.markdown('---')
st.markdown('## ▶️ Probabilidades dos Jogos da Copa')
jogoscopa = pd.read_excel('estimativasJogosCopa.xlsx', index_col=0)

st.table(jogoscopa[['grupo', 'seleção1', 'seleção2', 'Vitória', 'Empate', 'Derrota']])