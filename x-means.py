#K-means
import pandas as pd
import numpy as np
import time
import datetime
import math
import scipy
 
#funcao pra calcular a distancia euclidiana entre dois vetores
def distanciaEuclidiana(vetor1, vetor2):
    tam = 0
    quar_distance = 0
    # Se vetor for multidimensional
    if(vetor1.ndim > 1):
        # linhas do vetor
        lin = vetor1.shape[0]
        # colunas do vetor
        col = vetor1.shape[1]
        # Enquanto houver cluster
        while tam < lin:
            index = 0
            # Enquanto houver dado
            while index < col:
                    # Soma a distancia entre os dados ao quadrado
                    quar_distance += (vetor1[tam][index] - vetor2[tam][index]) ** 2
                    index = index + 1
            tam = tam+1
    # Se o vetor for unidimensional
    else:
        # Quantidade de dados do vetor
        col = len(vetor1)
        index = 0
        # Enquanto houver dados
        while index < col:
            # Soma a distancia entre os dados ao quadrado
            quar_distance += (vetor1[index] - vetor2[index]) ** 2
            index = index + 1
        tam = tam+1
    # Retorna a raiz quadrada da soma, que e a distancia euclidiana    
    return math.sqrt(quar_distance)
   
#Como funciona o silhouette:
#Primeiro precisamos calcular a distancia media de cada dado i para todos os dados que estao no mesmo cluster
#esse vetor vai guardar a distância media de i para todos os dados que estao no mesmo cluster
#essa distancia sera calculada mais pra frente para todos os dados do conjunto de dados (no caso, de cada documento)
#o que acontece aqui e que veremos se cada documento esta bem ajustado em cada cluster
#implementacao do Silhouette para kmeans com distancia euclidiana
#para implementarmos o silhoutte
def silhouette (vetorNumeros, dados_cluster, numCluster):  
   
    #calculamos o total de dados que vamos ter que iterar
    totalDocs = vetorNumeros.shape[0]
   
    #criamos um vetor pra guardar quantos dados temos em cada cluster
    totalCluster = [0]*numCluster
   
    #iterador
    i = 0
   
    #criamos o vetor silhouette que vai guardar o valor de S pra cada dado
    #s(i) = (b(i) - a(i))))/(max{a(i), b(i})
    silhouette = [0]*totalDocs
   
    #vamos fazer a contagem de quantos elementos estao em cada cluster
    while i < totalDocs:
       
        #para cada documento que temos vamos verificar a qual cluster ele pertence
        #quando encontramos o cluster correto, iteramos em +1 a quantidade de documentos que pertencem a aquele cluster
        totalCluster[int(dados_cluster[i])] += 1
       
        i += 1
   
    dado = 0
   
    #criamos o vetor A que vai guardar o valor final da distancia media de cada dado para todos os dados que estao no mesmo
    #cluster que ele
    A = [0]*totalDocs  
   
    #criamos um vetor com numDocs como numero de linhas porque vamos calcular a distancia media dos dados do mesmo cluster para
    #cada um
    tempA = [0]*totalDocs
   
    #criamos o vetorB que vai guardar a distancia minima media entre o dado e todos os dados de cada cluster a que ele nao pertence
    B = [0]*totalDocs
   
    #criamos tambem um vetor temporario para ajudar nessas operacoes
    tempB = np.zeros((totalDocs, numCluster))
   
    #nesse laco vamos fazer a construcao do vetor S (silhouette)
    #para isso vamos passar por cada dado para calcular duas distancias
    #a primeira distancia e o calculo da distancia media do dado i a todos os outros dados que estao no mesmo cluster que ele
    #a segunda distancia e o calcula da minima distancia media do dado i a todos os outros dados que estao nos clusters em que
    #ele nao esta
    while dado < totalDocs:
       
        #Calculos das distancias
       
        #iterador
        j = 0
       
        #primeiro: preciso saber se os dados pertencem ao mesmo cluster:
        #se sim, somo a distancia desses dados referente ao dado que eu estou trabalhando
        #se não, preciso calcular a distancia do dado i pra todos os dados de todos os outros clusters
        #e depois calcular a menor distancia media
        while j < totalDocs:
           
            if(dado != j and (int(dados_cluster[dado]) == int(dados_cluster[j]))):
               
                #significa que temos que calcular as distancias de todos os dados que estao no mesmo cluster    
               
                tempA[dado] += distanciaEuclidiana(vetorNumeros[dado], vetorNumeros[j])
               
            if(dado != j and (int(dados_cluster[dado]) != int(dados_cluster[j]))):
               
                #siginifica que temos que calcular a distancia do dados pra todos os outros dados dos outros clusters
                #e eleger a menor distancia media como "campea"
               
                tempB[dado][int(dados_cluster[j])] += distanciaEuclidiana(vetorNumeros[dado], vetorNumeros[j])
               
            j += 1
       
        #se o valor final de A para um determinado dado for igual a 0, nem fazemos conta
        if(tempA[dado] != 0):
   
            A[dado] = tempA[dado]/totalCluster[int(dados_cluster[dado])]
       
        k = dado
        l = 0
       
        #pra cada dado, vamos iterar para arrumar o vetor B
        #(iteramos de acordo com o total de clusters, que é a mesma quantidade de colunas que temos em tempB)
        while l < numCluster:
           
            #se tempB do dado i no cluster c for 0, significa que eles pertencem ao mesmo cluster, entao essas contas ja foram
            #feitas em A
            #nesse caso, para nao atrapalhar depois em que vamos tirar o minimo das distancias medias, colocamos um valor bem
            #alto para esse valor nunca ser pego e nao atrapalhar as contas
            if(tempB[k][l] == 0):
               
                tempB[k][l] = 9999
           
            else:
               
                #se existe valor, entao atualizamos o valor para a distancia media entre o dado e os dados de cada cluster
                tempB[k][l] = tempB[k][l]/totalCluster[l]
               
            l += 1
       
        #pegamos o minimo da distancia media
        B[dado] = np.min(tempB[dado])
       
        #Construcao do vetor Silhouette
        #fazer atualizacao do silhouette
        silhouette[dado] = (B[dado]-A[dado])/(max(A[dado],B[dado]))        
       
        dado += 1
   
    return silhouette
   
def inicializeX(max_clusters, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,vetorPalavras):
        #executa o kmeans e depois verifica se a quantidade de clusters para essa iteracao
        #eh boa ou nao
        s_ant = -9999999
        # numero k agrupamentos ideal a ser ajustado
        clusters_ideal = 0
        for clusters in range(1,max_clusters):
            # Zerar as variaveis
            centroid_list = None
            centroides = None
            dados_cluster = None
            # Centroid_list estava fora do for, mas tem que estar dentro para sempre ajustar centroides de acordo com o numero de clusters
            centroid_list = vetor_dados[np.random.randint(0, num_linhas - 1, size=clusters)]
            centroides, dados_cluster, iteracoes, num_clusters = kmeans(clusters, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,centroid_list)
            print("NUM_CLUSTERS: "+str(clusters))
            S = silhouette(vetor_dados, dados_cluster, num_clusters)
            S = S.sum()
            print("S: "+str(S))
            print("s_ant: "+str(s_ant))
            if (S > s_ant):
                clusters_ideal = num_clusters
                s_ant = S
        print("CLUSTER IDEAL"+str(clusters_ideal))
        return clusters_ideal
       
       # centroides, dados_cluster, iteracoes, num_clusters = [kmeans(k, taxa_erro, num_linhas, num_colunas, vetor_dados, max_iteracoes,centroid_list) for k in range(1,max_clusters)]
        #bic = [compute_bic(kmeansi.centroids,kmeansi.num_cluster_xmeans,vetor_dados,num_colunas) for kmeansi in result_kmeans]
 
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
       
    #Caso o vetor venha do plus ou x, precisamos transforma-lo num array compativel com a
    #biblioteca numpy para utilizar o np.zeros
    centroides = np.array(centroides)
       
    centroides_antigos = np.zeros(centroides.shape)
    #essa funcao faz as execucoes
    #sao varias chamadas feitas ao k-means, variando os parametros numClusters, taxa_erro e iteracoes
   
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
    while iteracoes < max_iteracoes:
         
        if(distancia_centroides < taxa_erro):
            break
       
        #a iteracao aumenta em 1
        iteracoes += 1
       
        #recalculamos a distancia
        #print("CENTROIDES:" +str(centroides))
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
 
    return centroides, dados_cluster, iteracoes, num_clusters

arquivo = 'C:/Users/Mariana/Desktop/EP IA - Sarajane/Saidas_teste/Process_Mining_Abstracts_TF.xlsx'
#arquivo = 'C:/Users/Mariana/Desktop/IA/Saidas_teste/Process_Mining_Abstracts_Binario.xlsx'
#arquivo = 'C:/Users/Mariana/Desktop/IA/Saidas_teste/TFIDF.xlsx'
 
#a matriz dados vai guardar todo o conteudo lido do arquivo, inclusive os cabecalhos (documentos e palavras)
dados = pegaTexto(arquivo)
 
#o vetor de palavras vai guardar o cabecalho "linha" que foi lido do Excel
palavras = dados.axes[1].tolist()
 
#o vetor de documentos vai guardar o cabecalho "coluna" que foi lido do Excel
documentos = dados.axes[0].tolist()
 
#a variavel "colunas" guarda a quantidade de colunas que existem no excel passado
colunas = dados.shape[1]
 
#a variavel "linhas" guarda a quantidade de linhas que existem no arquivo passado
linhas = dados.shape[0]
 
#o vetor conteudo guarda os valores numericos pegos no excel, ou seja, as aparicoes das palavras nos textos
conteudo = dados.values.tolist()
 
#esse vetor "valores" vai guardar todas as informacoes numericas que sao pegas no excel
#ou seja, esse vetor vai conter as aparicoes das palavras nos textos
valores = np.zeros((linhas, colunas))
 
a = 0
b = 0
 
#esse loop vai guardar os valores de conteudo em um outro vetor
while a < linhas:
    while b< colunas:
        valores[a][b] = conteudo[a][b]
        b+=1
    b = 1
    a+=1
 
 
#funcao que faz a escrita do log execucao
def escrita(arquivo_entrada, num_clusters, tempo_total, erro, docs, vetor_cluster, numArquivo, iteracoes):
   
    #essa funcao faz a escrita do log de execucao de cada chamada ao k-means e salva no seguinte formato:
    arquivo = open('C:/Users/Mariana/Desktop/EP IA - Sarajane/saidas/execucao'+str(numArquivo)+'.txt', 'w')
   
    now = datetime.datetime.now()
   
    arquivo.write('Execução feita em '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+' - '+str(now.hour)+':'+str(now.minute)+'\n')
   
    arquivo.write('\n')
    arquivo.write('Arquivo de entrada '+arquivo_entrada+'\n')
    arquivo.write('Numero de clusters: '+str(num_clusters)+'\n')
    arquivo.write('Tempo total de execução: '+str(tempo_total)+'\n')
    arquivo.write('Taxa de erro máxima permitida: '+str(erro)+'\n')
    arquivo.write('Total de iteracoes: '+str(iteracoes)+'\n')
    arquivo.write('Distância usada: euclidiana')
    arquivo.write('\n')
    arquivo.write('\n')
   
    #mandar qual documento ficou em cada cluster
    m = 0
   
    while m < len(docs):
        arquivo.write('Documento '+str(docs[m])+' - Cluster '+str(int(vetor_cluster.item(m)))+'\n')
        m += 1
   
   
    arquivo.close()
   
    return    
 
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
 
def encontraDadosPalavras(palavras):
       
    #escrita(entrada_dados, )
 
    #vamos pegar tudo o que esta no dataframe lido e salvar numa lista
    texto_quebrado = palavras.values.tolist()
   
    #descobrimos o tamanho dessa matriz que veio do dataframe
    linhas = len(texto_quebrado)
    colunas = len(texto_quebrado[0])
   
    #temos o dataframe quebrado numa grande matriz
    #agora separamos, na matriz, o que e dado do que e palavra
    palavras = []
   
    #e criamos um vetor que vai guardar todos os numeros que estao associados a uma palavra
    vetorNumeros = np.zeros((linhas, colunas - 1))
       
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
            vetorNumeros[a][b - 1] = texto_quebrado[a][b]
            b+=1
        b = 1
        a+=1
   
    return palavras, vetorNumeros
 
def execute(arquivo_entrada, palavras, num_clusters, erro, iteracoes, max_clusters):
    #essa funcao faz as execucoes
    #sao varias chamadas feitas ao k-means, variando os parametros numClusters, taxa_erro e iteracoes
 
    vetorPalavras, vetorNumeros = encontraDadosPalavras(palavras)
    num_linhas = vetorNumeros.shape[0]
    num_colunas = vetorNumeros.shape[1]    
   
    start = time.time()
    centroid_list = []
    num_clusters = inicializeX(max_clusters, erro,num_linhas, num_colunas, vetorNumeros, iteracoes,vetorPalavras)
    centroid_list = vetorNumeros[np.random.randint(0, num_linhas - 1, size=num_clusters)]
    centroides, dados_cluster, iteracoes, num_cluster_xmeans = kmeans(num_clusters, erro, num_linhas, num_colunas, vetorNumeros, iteracoes,centroid_list)
    tempo_total = ("--- %s seconds ---" % (time.time() - start))
 
    #escrita(entrada, numClusters, tempo_total, maxErro, vetorPalavras, dados_cluster, 1, iteracoes)
    result = silhouette(vetorNumeros, dados_cluster, num_clusters)
    return result
 
def main():
   
    max_clusters = 11
    num_clusters =2
    erro = 0.2
    iteracoes = 30
    arquivo_entrada = '/home/juny/InteligenciaArtificial/pre_process/Saida/TFIDF.csv'
   
    #pegamos o csv lido e transformamos num dataframe
    #no entanto nao podemos fazer manipulacao de palavras e nem de numeros nele
    palavras = pegaTexto(arquivo_entrada)
   
    # Remove primeira coluna (inutilizavel)
    palavras = palavras.drop(palavras.columns[[0]], axis=1)    
   
    # Num total de 127650, escolhemos um conjunto de 25000
    palavras = palavras.iloc[:,:25000]    
   
    # Array que ira armazenar a soma total para cada palavra
    sumArray = np.zeros(shape=(palavras.shape[1]))
   
    # Array com os valores armazenados, podemos ver o mínimo de palavras, o máximo e estipular um valor para remover palavras "desnecessárias"
    sumArray = contaPalavras(palavras, sumArray)    
   
    # Remove palavras 'desnecessarias' conforme parametros
    palavras = removePalavras(palavras)
   
    y = execute(arquivo_entrada, palavras, num_clusters, erro, iteracoes, max_clusters)
   
    print(y)
   
    return