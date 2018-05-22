import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.__version__
import sys
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.umatrix import UMatrixView
import sompy
#funcao que carrega o arquivo de entrada para kmeans
def carregaTXT(nome):    
    return np.loadtxt(nome)
 
#funcao usada pra pegar as palavras que existem no csv
def pegaTexto (nome):  
    return pd.read_csv(nome, encoding = 'iso-8859-1')
 

# Funcao para contar o total de palavras, afim de saber palavras que podemos filtrar
def contaPalavras(palavras, sumArray):
    # inicializando variaveis
    col = 0
    i = 0
    # Total de colunas no dataframe
    col = len(palavras.columns)
    # loop para armazenar no vetor o somatorio de cada palavra
    while i < col:
        # somatorio da coluna 'i'
        soma = sum(palavras.iloc[:,i])    
        # armazena na posicao 'i' do vetor o somatorio da coluna 'i'
        sumArray[i] = soma
        i = i+1
   
    return sumArray
 
 
def removePalavras(palavras):
   
    i = 0
    # Total de colunas no dataframe
    i = len(palavras.columns)
    # 0-9 = 10 elementos (subtrai 1 para nao dar index array out of bounds)
    i = i-1
    # Ajustar de acordo com dataFrame o min e o max do vetor total de palavras
    min = 40
    max = 140
       
    while i > 0:
        soma = sum(palavras.iloc[:,i])      
        if (soma < min or soma > max):      
            palavras = palavras.drop(palavras.columns[[i]], axis=1)
        i = i-1
   
    return palavras
 

 
def main():
   
    arquivo_entrada = 'C:/devel/InteligenciaArtificial/TF.csv'
   
    #pegamos o csv lido e transformamos num dataframe
    #no entanto nao podemos fazer manipulacao de palavras e nem de numeros nele
    palavras = pegaTexto(arquivo_entrada)

    # Remove primeira coluna (inutilizavel)
    palavras = palavras.drop(palavras.columns[[0]], axis=1)    
   
    # Num total de 127650, escolhemos um conjunto de 25000
    palavras = palavras.iloc[:,:7000]    
   
    # Array que ira armazenar a soma total para cada palavra
    #sumArray = np.zeros(shape=(palavras.shape[1]))
   
    # Array com os valores armazenados, podemos ver o mínimo de palavras, o máximo e estipular um valor para remover palavras "desnecessárias"
    #sumArray = contaPalavras(palavras, sumArray)    
   
    # Remove palavras 'desnecessarias' conforme parametros
    #palavras = removePalavras(palavras)
    
    palavras = palavras.as_matrix()
    
    #quando reclamar de 1-dimensao, usar esse codigo
    #palavras = np.array(palavras)
    
    # Tamanho do mapa utilizado no SOM - Quanto maior o número de dados, melhor colocar um número MxM maior para um cluster mais preciso
    msz0 = 6
    msz1 = 6   

    # Criação e treino do SOM
    sm = SOMFactory.build(palavras, mapsize = [msz0, msz1], initialization='pca')
    
    #Ver como colocar nomes
    #sm.component_names = features
    
    sm.train(n_job = 1, shared_memory = 'no',verbose='info')
    
    #quando reclamar de falta de cluster, usar esse codigo
    cl = sm.cluster(n_clusters=4)
    
    # Visualization class
    view2D  = View2D(6,6,"Data Map",text_size=10)
    hitmap  = HitMapView(15,15,"Cluster Hit Map",text_size=10)
    umat  = UMatrixView(100,100,"Unified Distance Matrix", text_size=14)

    # Pede pra que a visualização seja em 1 dimensao, entao nao está funcionando no momento
    #view2D.show(sm, col_sz=3, desnormalize=True)
    # O mesmo caso do de cima, tem que ver 
    #hitmap.show(sm)
    #umat.show(sm)
    
    #codigo de visualização com numeração dos clusteres
    h  =  sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
    h.show(sm)
    
# =============================================================================
#     v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)  
#     v.show(sm, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
# =============================================================================


    return
main()