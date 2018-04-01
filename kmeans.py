#SITES UTILIZADOS COMO BASE:
#https://gist.github.com/iandanforth/5862470
#https://juliocprocha.wordpress.com/2017/06/12/k-means-em-python/


from math import sqrt
import random

def main():
    #numero de categorias identificadas no .csv
    numeroCategorias = 20
    
    #numero de dimensões dos dados do arquivo .csv
    numeroDimensoes = 2
    
    #Intervalo de valores minimos e maximos em cada categoria
    minCategoria = 0
    maxCategoria = 30
    
    #Numero de clusters que serão formados
    numeroClusters = 3
    
    #Valor minimo de taxa de erro para aceitar a  convergencia do algoritmo
    taxaErro = 0.5
    
    #Leitura dos dados do arquivo .csv 
    dados = leituraCSV()
    
    #Agrupamento pelo metodo k-means
    iteracaoCont = numeroCategorias
    agrupamento = analiseKmeans(dados,numeroClusters,taxaErro,iteracaoCont)

def analiseKmeans(dados,numeroClusters,taxaErro,iteracaoCont):
    #lista com os agrupamentos obtidos
    agrupamentosObtidos = []
    
    #lista com os erros dos agrupamentos obtidos
    erros = []
    
    #Executamos o k-means para cada categoria do .csv
    for i in range(iteracaoCont):
        agrupamentos = kmeans(dados,numeroClusters,taxaErro)
        errro = calculaErro(agrupamentos)
        
        #adiciona os valores obtidos nos vetores
        agrupamentoObtidos.append(agrupamentos)
        erros.append(erro)
        
    #Retorna o agrupamento com menor erro    
    menorErro = min(erros)
    return agrupamentosObtidos[menorErro]

#Metodo onde é feito o agrupamento
def kmeans(dados,numeroClusters, taxaErro):
    
    #aleatoriamente o vetor prototipo eh inicializado em cima de um dado
    vetoresPrototipos = random.sample(dados,numeroClusters)

    #todos os dados são colocados num vetor
    agrupamento = [Cluster([p])] for p in vetoresPrototipos

    #contador do numero de iteracoes ate o momento
    contador = 0
    while True:
        #cria uma matriz para armazenar os dados nos respectivos agrupamentos
        matriz = [[] for _ in agrupamento]
        numeroAgrupamento = len(agrupamento)

        loopCont += 1
        #operacao para todos os dados
        for dado in dados:
            ##primeiro atribuimos o dado para o primeiro agrupamento,
            ##assim  teremos uma distancia para comparar posteriormente.
            
            agrupamentoIndex = 0
            menorDistacia = distaciaEuclidiana(dado,agrupamento[0].centroid)

            for i in range(1,numeroAgrupamento):
                #calcula a distancia o dado até os outros vetores prototipos
                distancia = distaciaEuclidiana(dado,agrupamento[i].centroid) 
                #se o dado estiver mais proximo de outro vetor prototipo,
                #a variavel de menor distancia eh atualizada
                if distancia < menorDistacia:
                    menorDistacia = distancia
                    agrupamentoIndex = i
            #apos achar a qual vetor prototipo o dado pertence,
            #precisamos atualizar a nossa matriz que contem as 
            #informacoes de qual dado pertence a qual vetor
            matriz[agrupamentoIndex].append(dado)  

        #variavel utilizada para armazenar a taxa de erro dessa iteracao
        erroTotal = 0.0

        #calculando qual foi a taxa de erro de cada vetor prototipo
        for i in range(numeroAgrupamento):
            erro = agrupamento[i].update(matriz[i])
            #verifica entre todas as taxas de erro, qual foi a maior
            erroTotal = max(erroTotal,erro)
        #se o erroTotal for menor do que a taxa de erro considerada aceitavel
        #o algoritmo convergio
        if erroTotal < taxaErro:
            break   

    return agrupamento


    

#recebe dois vetores e devolve a distancia entre eles
def distEuclidiana(vetor1, vetor2):
    distanciaQuadrado = 0.0;
    #calcula a distancia ao quadrado
    for i in range(len(vetor1)):        
        distanciaQuadrado += pow((vetor1[i] - vetor2[i],2))
    
    #Tira a raiz quadrada da distancia
    distanciaFinal = sqrt(distanciaQuadrado)
    return distanciaFinal


#VER UM JEITO MELHOR DE IMPLEMENTAR ESSAS CLASSES
class Dados(object):
    def __init__(self, coordenadas):
        '''
        coords - A list of values, one per dimension
        '''

        self.coordenadas = coordenadas
        self.n = len(coordenadas)

    def __repr__(self):
        return str(self.coordenadas)


class Cluster(object):
    def __init__(self,dados):
        self.dados = dados
        self.centroid = self.calculaCentroid()

    def calculaCentroid(self):
        numDados = len(self.dados)
        coordenadas = [p.coordenadas for p in self.dados]
        unzipped = zip(*coordenadas)    
        centroid_coordenadas = [math.fsum(dList)/numDados ofr dList in unzipped]

        return Dados(centroid_coordenadas)
