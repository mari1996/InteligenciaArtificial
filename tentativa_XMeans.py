# -*- coding: utf-8 -*-

#*********************

#Usar como base para X-Means
#https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

#**********************

#K-means
import pandas as pd
import numpy as np
import time
import datetime
import scipy

#funcao pra calcular a distancia euclidiana entre dois vetores
def distanciaEuclidiana (vetor1, vetor2):   
    return np.linalg.norm(vetor1 - vetor2)

def silhouette (vetorNumeros, dados_cluster):
    
    # Definir o número máximo de clusters
    maxCluster = 2
    # Silhouette A armazena distancia para dados do mesmo cluster
    silhouetteA = []
    # Pega o total de dados
    linha = len(vetorNumeros)-1
    # Matriz que irá armazenar a distancia de um dado para todos os dados possiveis por cluster
    silhouetteB = np.zeros(shape=(linha,maxCluster))
    # Armazena quantos elementos há em cada cluster
    elementosCluster = np.zeros(shape=(maxCluster), dtype=int)
    
    # Iteramos para cada palavra
    for i in range(linha):
        j = 0
        # Distancia usada para elementos do mesmo cluster
        distancia = 0
        # Distancia usada para elementos de todos clusters
        distanciaExterna = 0
        # Cluster do vetor do dado 'i'
        clusterAtual = int(dados_cluster[i])
        # Incrementa número de dados no cluster
        elementosCluster[clusterAtual] +=  1 
        # Enquanto houver dado
        while j < linha: 
            # Não calcula distancia para o proprio dado
            if (i == j):
                pass
            # Pega o cluster correspondente ao dado 'j'
            clusterJ = int(dados_cluster[j])
            # Se o dado iterado estiver no mesmo cluster que o dado escolhido. Esse é o parametro 'Ai'
            if(clusterJ == clusterAtual):
                #Calcula a distancia do dado escolhido 'i' para o dado 'j' do mesmo cluster
                distancia = distancia + distanciaEuclidiana (vetorNumeros[j], vetorNumeros[i]) 
                # Armazena número alto para dado do mesmo cluster para invalidar dado do mesmo cluster e calcular Indice B depois
                silhouetteB[i][clusterJ] = -1
            # Se o dado iterado estiver em um cluster diferente do dado escolhido. Esse é o parametro 'Bi'
            else :
                # Calcula distancia do dado i para o dado j
                distanciaExterna = distanciaEuclidiana (vetorNumeros[j], vetorNumeros[i])
                # Armazena para o dado 'i' a soma da distancia para todos os dados do cluster 'j'
                silhouetteB[i][clusterJ] += distanciaExterna
            j = j + 1
            
        # Adiciona a distancia no parametro A[i]
        silhouetteA.append(distancia)
    #print(elementosCluster)  
    
    # Cria um vetor com a quantidade de dados para armazenar a metrica silhouette
    S = np.zeros(shape=linha)
    # Irá armazenar a menor distancia do parametro B
    for i in range(linha):
        B = 9999
        for k in range(maxCluster):
            # Armazena o menor valor
            if (silhouetteB[i][k] < B and silhouetteB[i][k] != -1):
                # Armazena a menor média de um dado 'i' para todos os dados de todos os cluster
                A = silhouetteA[i] / elementosCluster[int(dados_cluster[i])]
                B = (silhouetteB[i][k] / elementosCluster[k])
        S[i] = (distanciaEuclidiana(B,A))/max(B, A)
        
    return S

def compute_bic(centroids,number_clusters,data_points,label_clusters):
    # recebendo os centroids e label dos clusters
    centers = [centroids]
    labels  = label_clusters
    
    #numero de clusters
    m = number_clusters
    # tamanho de cada
    n = np.bincount(labels)
    # obtem a quantidade de dados utilizados
    N, d = data_points.shape
    cl_var = 0
    #compute variance for all clusters beforehand
    #VER SE A CONTA ESTA CERTA
    for i in range(1,m):
       # print("LABEL" + str(i))
       # print("parametro 1" + str(data_points[i]))
       # print("CENTERS" + str(centers))
       # print("parametro 2" + str(centers[0][i]))
       print("CONTA" + str(( (N - m))))
       cl_var = distanciaEuclidiana(data_points[i], [centers[0][i]])
       #cl_var = (1.0 / (N - m) / d) * sum([sum(distanciaEuclidiana(data_points[i], [centers[0][i]])**2)])
        
        
    const_term = 0.5 * m * np.log(N) * (d+1)

    bic = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(1,m)]) - const_term

    return(bic)
    
def inicializeX(max_clusters, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,vetorPalavras):
        #executa o kmeans e depois verifica se a quantidade de clusters para essa iteracao
        #eh boa ou nao
        centroid_list = vetor_dados[np.random.randint(0, num_linhas - 1, size=max_clusters)]
        for clusters in range(1,max_clusters):
            centroides, dados_cluster, iteracoes, num_clusters = kmeans(clusters, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,centroid_list)
            bic = compute_bic(centroides,num_clusters,vetor_dados,vetorPalavras)
            print("BIC" + str(bic))
        
        
       # centroides, dados_cluster, iteracoes, num_clusters = [kmeans(k, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,centroid_list) for k in range(1,max_clusters)]
        #bic = [compute_bic(kmeansi.centroids,kmeansi.num_cluster_xmeans,vetor_dados,num_colunas) for kmeansi in result_kmeans]
        
def inicializeNormal(vetor_dados,num_linhas,num_clusters ):
    #Retorna um centroid aleatorio
    return vetor_dados[np.random.randint(0, num_linhas - 1, size=num_clusters)]

def inicializePlus(vetor_dados,num_linhas,num_clusters):
    #Esse metodo busca inicializar os centroids com melhor distribuicao pelos dados
    #Para isso ele calcula a distancia dos dados ate os centroids ja existentes e 
    #tenta criar novos centroids de forma que todos os centroids fiquem uniformemente
    #distribuidos pelos dados
    
    #Cria um indice aleatorio que sera utilizado para escolher o centroid logo em seguida
    indiceInicial = np.random.randint(0, num_linhas)
    #Retorna um centroid aleatorio para iniciar o algoritmo
    centroids = [vetor_dados[indiceInicial]]
    for k in range(1, num_clusters):
        #Calculamos a menor distancia de cada dado para todos os centroids
        distancia = scipy.array([min([scipy.inner(c-x,c-x) for c in centroids]) for x in vetor_dados])
        #Calcula a probabilidade de escolher um dado como novo centroid
        probabilidade = distancia/distancia.sum()
        totalProbabilidade = probabilidade.cumsum()
        r = scipy.rand()
        #Se o numero gerado pela variavel 'r' for menor do que o valor de probabilidade
        #armazenado na variavel 'p', entao o novo centroid eh o dado na variavel 'p'
        for j,p in enumerate(totalProbabilidade):
            if r < p:
                i = j
                break
        centroids.append(vetor_dados[i])
    return centroids

#funcao que carrega o arquivo de entrada para kmeans
def carregaTXT(nome):    
    return np.loadtxt(nome)

#funcao usada pra pegar as palavras que existem no csv
def pegaTexto (nome):   
    return pd.read_csv(nome, encoding = 'iso-8859-1')

#funcao kmeans
def kmeans(num_clusters, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,centroid_list):   
    #decidir se vai receber o numero de clusters como parametro ou se vai usar max cluster
    #decidir se vai receber a taxa de erro como parametro ou se vai usar a taxa de erro definida fora
   
    #criamos uma lista que vai guardar os centroides 
    #esses centroides vao ser inicializados de forma aleatoria
    centroides = centroid_list
    
    #para podermos calcular o erro medio precisamos comparar a lista de centroides atuais com a lista anterior de centroides
    #essa lista é definida com o mesmo tamanho que a nossa lista de centroides
    #inicializamos esse vetor com 0s
    
    centroides_antigos = np.zeros(centroides.shape)
    
    #como cada dado vai pertencer a um cluster diferente, criamos uma lista que vai conter todos os dados e em quais
    #clusters eles vao estar
    #inicializamos esse vetor com 0s
    #vetor de dados x cluster
    dados_cluster = np.zeros((num_linhas, 1))
    
    #precisamos calcular a distancia entre os dois vetores de centroides, o antigo e o novo, para saber se diminuimos o erro ou nao
    #calculamos essa distancia usando a distancia euclidiana
    #se a distancia (ou erro) for menor que a taxa de erro passada, significa que nao precisamos mais calcular novos centroides
    distancia_centroides = distanciaEuclidiana(centroides, centroides_antigos)
    
    #guardamos o número de iterações pra não passarmos do máximo de iteracoes permitidas
    iteracoes = 0
    
    #vamos recalcular os centroides apenas se o erro ainda for maior do que a taxa permitida e se o numero maximo de iteracoes ainda 
    #nao foi alcancado
    #verificar se é or ou and aqui
    while iteracoes < max_iteracoes and distancia_centroides > taxa_erro:
        
        #a iteracao aumenta em 1
        iteracoes += 1
        
        #recalculamos a distancia
        distancia_centroides = distanciaEuclidiana(centroides, centroides_antigos)
        
        #como estamos numa nova iteracao, consideramos que os centroides atuais agora vao ser antigos
        #porque vamos recalcular a lista de centroides
        centroides_antigos = centroides
        
        #como vamos calcular os novos centroides precisamos saber a distancia de cada dado para cada centroide atual
        #para isso, criamos um vetor de distancias que vai guardar essas informacoes
        #a funcao enumerate enumera os dados que estamos vendo, nesse caso, os dados de vetor_dados
        #nesse caso, cada dado tera um indice proprio
        for indice_dado, dado in enumerate(vetor_dados):
            #definimos um vetor de tamanho num_clusters, pois temos num_clusters centroides que devem ser recalculados
            #distancia_dados = np.zeros((num_clusters, 1))
            
            #ADICIONEI A LINHA SEGUINTE
            tamanho = len(centroides)
            distancia_dados = np.zeros((tamanho,1))
            #preenche vetor com -1
            #distancia_dados.fill(-1)
            #para cada centroide que possuimos, precisamos calcular as distancias dos dados ate eles
            for indice_centroide, centroide in enumerate(centroides):
                #vamos calcular a distancia de cada dado ate cada centroide
                #as menores distancias vao mostrar onde os dados devem ficar
                distancia_dados[indice_centroide] = distanciaEuclidiana(centroide, dado)
                #essas distancias estao armazenadas num vetor distancia de dados X centroide
            #queremos identificar em qual cluster cada dado vai ficar
            #para isso, criamos uma lista contendo todos os dados e a qual cluster ele pertence (nessa iteracao)
            #dizemos que o dado X esta no cluster em que a distancia entre ele e o centroide N e a menor
            #a funcao np.argmin retorna o menor valor
            dados_cluster[indice_dado, 0] = np.argmin(distancia_dados)
        #criamos uma lista de centroides temporarios que sao atualizados dentro dessa iteracao para nao perdermos o que 
        #estamos calculando
        
        #ADICIONEI A LINHA SEGUINTE
        len_centroid = len(centroides)
        temp_centroides = np.zeros((len_centroid, num_colunas))
        
        #para cada cluster vamos encontrar todos os dados que estao mais proximos dele e vamos achar o ponto medio entre 
        #eles
        #esse ponto medio sera o valor do novo centroide
        for indice in range(len(centroides)):
            dados_proximos = [i for i in range(len(dados_cluster)) if dados_cluster[i] == indice]   
            
            #se o valor da distancia for zero, definir esse centroide novo como o centroide antigo
            
            #tratamos o caso do cluster ainda não ter dados próximos a ele
            #se o cluster ainda não possui dados proximos a ele, pulamos esse caso
            if dados_proximos == []:
                break
            
            #encontramos o valor medio desses dados proximos, que vai ser o nosso novo centroide
            #adicionar um if pra não entrar no caso de ser 0
            centroide = np.mean(vetor_dados[dados_proximos], axis=0)
            #acrescentamos o novo centroide na lista de centroides temporarios
            temp_centroides[indice, :] = centroide
            
        centroides = temp_centroides
#           
    return centroides, dados_cluster, iteracoes, num_clusters

def encontraDadosPalavras(arquivo_entrada):
       
    #escrita(entrada_dados, )
    
    #pegamos o csv lido e transformamos num dataframe
    #no entanto nao podemos fazer manipulacao de palavras e nem de numeros nele
    palavras = pegaTexto(arquivo_entrada)
    
    #vamos pegar tudo o que esta no dataframe lido e salvar numa lista
    texto_quebrado = palavras.values.tolist()
    
    #descobrimos o tamanho dessa matriz que veio do dataframe
    linhas = len(texto_quebrado)
    colunas = len(texto_quebrado[0])
    
    #temos o dataframe quebrado numa grande matriz
    #agora separamos, na matriz, o que e dado do que e palavra
    palavras = []
    
    #e criamos um vetor que vai guardar todos os numeros que estao associados a uma palavra
    valores_palavras = np.zeros((linhas, colunas - 1))
       
    #esse laço vai percorrer a lista que representa o dataframe, pegar todas as palavras e salvar numa lista separada
    i = 0

    while i < linhas:
        palavras.append(texto_quebrado[i][0])
        i += 1

    #esse laco vai percorrer a lista que representa o dataframe e pegar os valores que estao associados a cada palavra
    a = 0
    b = 1
    
    while a < linhas:
        while b < colunas - 1:
            valores_palavras[a][b - 1] = texto_quebrado[a][b]
            b+=1
        b = 1
        a+=1
    
    return palavras, valores_palavras


#funcao que faz a escrita do log execucao
def escrita(arquivo_entrada, num_clusters, tempo_total, erro, vetor_palavras, vetor_cluster, numArquivo, iteracoes):
    
    #essa funcao faz a escrita do log de execucao de cada chamada ao k-means e salva no seguinte formato:
    #arquivo de saida:
        #arquivo de entrada usado
        #numero de clusters
        #tempo de execucao
        #taxa de erro para aquela execucao
        #quantidade de dados por cluster (descobrir um jeito de fazer um while que vai contar quantos clusters
            #tem e jogar numa lista, e então vai sair contando quantos elementos tem em cada cluster)
        #dados x cluster
    arquivo = open('C:/Users/felip/Desktop/IA'+str(numArquivo)+'.txt', 'w')
    
    now = datetime.datetime.now()
    
    arquivo.write('Execução feita em: '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+' - '+str(now.hour)+':'+str(now.minute)+'\n')
    
    arquivo.write('\n')
    arquivo.write('Arquivo de entrada: '+arquivo_entrada+'\n')
    arquivo.write('Numero de clusters: '+str(num_clusters)+'\n')
    arquivo.write('Tempo total de execução: '+str(tempo_total)+'\n')
    arquivo.write('Taxa de erro permitida: '+str(erro)+'\n')
    arquivo.write('Total de iteracoes: '+str(iteracoes)+'\n')
    arquivo.write('Distância usada: euclidiana')
    arquivo.write('\n')
    arquivo.write('\n')
    
    #escrever o total de dados por cluster
    
    #printar os dados e os clusters
    m = 0
    
    while m < len(vetor_palavras):
        arquivo.write('Palavra: '+str(vetor_palavras[m])+' - Cluster: '+str(int(vetor_cluster.item(m)))+'\n')
        m += 1
    
    arquivo.close()
    
    return

def execute(entrada, maxClusters, maxErro, maxIters, typeKMeans):
    #essa funcao faz as execucoes 
    #sao varias chamadas feitas ao k-means, variando os parametros numClusters, taxa_erro e iteracoes
    max_clusters = 5  


    vetorPalavras, vetorNumeros = encontraDadosPalavras(entrada)
    
    start = time.time()
    centroid_list = []
    if(typeKMeans == 'normal'):
       centroid_list = inicializeNormal(vetorNumeros,vetorNumeros.data.shape[0],maxClusters)
    if(typeKMeans == 'plus'):
       centroid_list = inicializePlus((vetorNumeros,vetorNumeros.data.shape[0],maxClusters))
    if(typeKMeans == 'x'):  
        maxClusters = inicializeX(max_clusters, erro,vetorNumeros.data.shape[0], vetorNumeros.data.shape[1], vetorNumeros, maxIters,vetorPalavras)
        centroid_list = vetorNumeros[np.random.randint(0, vetorNumeros.data.shape[0] - 1, size=maxClusters)]
    
    centroides, dados_cluster, iteracoes, num_cluster_xmeans = kmeans(maxClusters, erro, vetorNumeros.data.shape[0], vetorNumeros.data.shape[1], vetorNumeros, maxIters,centroid_list)
    tempo_total = ("--- %s seconds ---" % (time.time() - start))

     
    escrita(entrada, maxClusters, tempo_total, maxErro, vetorPalavras, dados_cluster, 1, iteracoes)

    result = silhouette(vetorNumeros, dados_cluster)
    return result


arquivo = 'C:/Users/felip/Desktop/IA/teste.csv'


num_clusters = 2
erro = 0.5
iteracoes = 10

y = execute(arquivo, num_clusters, erro, iteracoes,'x')
print(y)
