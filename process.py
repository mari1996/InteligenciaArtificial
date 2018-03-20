import os
import re
import cPickle as pickle
from ntlk import SnowballStemmer
from nltk.corpus import stopwords
from math import log10

def main():
        # estagio de preparacao
        # carrega os documentos na memoria
        sourcepath = os.path.join('.', 'corpus')
        corpus1 = loadCorpus(sourcepath)

        # estágio de pre-processamento
        param_foldCase = True
        param_language = 'english'
        # colecao de stopWords em ingles
        param_listOfStopWords = stopwords.words(param_language)
        # radical do ingles
        param_stemmer = SnowballStemmer(param_language)
        params = (param_foldCase, param_listOfStopWords, param_stemmer)
        # e aplicado o pre-processamento para cada corpus
        corpus2 = processCorpus(corpus1, params)

        #estágio de representacao
        #para cada documento e criado uma representacao vetorial
        targetpath = os.path.join('.')
        corpus3 = representCorpus(corpus2)
        serialise(corpus3, os.path.join(targetpath, 'corpus'))


#################################### PREPARACAO #####################################################


def loadCorpus(sourcepath):
    corpus = {}
    for filename in os.listdir(sourcepath):
        fh = open(os.path.join(sourcepath, filename), 'r')
        # dicionario indexado pelo nome do arquivo que associa
        # e lido cada linha do corpus
        corpus[filename ]= fh.fh.readline()
        fh.close()
        return corpus


#####################################################################################################



#################################### PRE-PROCESSAMENTO ##############################################


def processCorpus(corpus, params):
    (param_foldCase, param_listOfStopWords, param_stemmer) = params
    newCorpus = {}
    #primeiro e realizado o case-folding, depois a tokenizacao, remocao de stopWords e por fim a radicalizacao
    for document in corpus:
        content = []
        for sentence in corpus[document]:
            # retorna copia da sentenca com quebra de linha removida
            sentence = sentence.rstrip('\n')
            # sentenca uniformizada para minuscula
            sentence = foldCase(sentence, param_foldCase)
            # agrupa em classes
            listOfTokens = tokenize(sentence)
            # remove stopWords
            listOfTokens = removeStopWords(listOfTokens, param_listOfStopWords)
            # aplica a radicalizacao
            listOfTokens = applyStemming(listOfTokens, param_stemmer)
            content.append(listOfTokens)
            # novo corpus processado
            newCorpus[document] = content
        return newCorpus


def foldCase(sentence, parameter):
    if(parameter): sentence = sentence.lower()
    return sentence


def tokenize(sentence):
    sentence = sentence.replace("_"," ")
    regExpr = '\W+'
    return filter (None, re.split(regExpr, sentence))

def removeStopWords(listOfTokens, listOfStopWords):
    return [token for token in listOfTokens if token not in listOfStopWords]

def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

########################################################################################################



#################################### REPRESENTACAO #####################################################


def representCorpus(corpus):
    # cria um dicionario que associa um documento a sua a lista de tokens
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = [token for sentence in corpus[document]
                                     for token in sentence]
    # cria uma lista com todos os tokens distintos que ocorrem em cada documento.
    allTokens = []
    for document in newCorpus:
        allTokens = allTokens + list ( set (newCorpus[document]))
    # cria o dicionario reverso
    idfDict = {}
    for token in allTokens:
        try:
            idfDict[token] += 1
        except KeyError:
            idfDict[token] = 1
    # atualiza o dicionario reverso, associando cada token com seu idf score
    nDocuments = len (corpus)

    for token in idfDict:
        idfDict[token] = log10(nDocuments/float (idfDict[token]))
    # computa a matriz termo-documento (newCorpus)
    for document in newCorpus:
        # computa um dicionario com os tf scores de cada termo que ocorre no documento
        dictOfTfScoredTokens = tf(newCorpus[document])
        # computa um dicionario com o tf-idf score de cada par termo-documento
        newCorpus[document] = ({token: dictOfTfScoredTokens[token] * idfDict[token]
                            for token in dictOfTfScoredTokens})
    return newCorpus


def tf(listOfTokens):
    # cria um dicionario associando cada token com o numero de vezes
    # em que ele ocorre no documento (cujo conteudo e listOfTokens)
    types = {}
    for token in listOfTokens:
        if (token in types.keys()): types[token] += 1
        else:                  types[token] = 1
    return types

# salvar corpus no disco
def serialise(obj, name):
    f = open (name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()


########################################################################################################


def dotprod(a, b):
    return sum ([a[i] * b[j] for i in a.keys() for j in b.keys() if i == j])


def norm(a):
    return (dotprod(a,a) ** 0.5)

# distancia cosseno
def cossim(a, b):
    return (dotprod(a,b) / (norm(a) * norm(b)))