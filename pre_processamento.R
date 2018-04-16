#representacoes por frequencia, binaria e tfidf

#e necessario rodar os comandos abaixo antes de continuar, para que as bibliotecas usadas sejam instaladas (se ainda 
#nao estiverem instaladas)
#install.packages('tm')
#install.packages('SnowballC')
#install.packages('stringr')

#biblioteca usada especificamente para text-mining
library(tm)
#biblioteca que disponibiliza o uso do stemming (reducao de palavras para o seu radical)
library(SnowballC)
#biblioteca que disponibiliza funcoes para manipualacao de strings
library(stringr)

#captura o diretorio
#Nesse caso, para rodar, e necessario alterar o diretorio para o local onde estao os textos no seu computador
#Pega todos os textos que estao nesse diretorio e transforma num corpus
CorpusVar <- Corpus(DirSource("C:/Users/Mariana/Desktop/IA/Textos/Entrada1/Olar"))


#deixa todas as letras em minusculo
CorpusVar <- tm_map(CorpusVar, content_transformer(tolower))

#remove pontuacao dos textos
CorpusVar <- tm_map(CorpusVar, removePunctuation)

#remove os numeros dos textos
CorpusVar <- tm_map(CorpusVar, removeNumbers)

#retira as stop words (lingua inglesa)
CorpusVar <- tm_map(CorpusVar, removeWords, stopwords("english"))

#remove espacos em branco que sao desnecessarios
CorpusVar <- tm_map(CorpusVar, stripWhitespace)

#reduz as palavras para o seu radical (lingua inglesa)
CorpusVar <- tm_map(CorpusVar, stemDocument, language="english")

#A primeira representacao e a por frequencia
#transforma o corpus em uma matriz de frequencia
tdm <- TermDocumentMatrix(CorpusVar)
#converte pra matriz
m <- as.matrix(tdm)
transp <- t(m)
write.csv(transp, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/Process_Mining_Abstracts_TF.csv")

#A segunda representacao e a binaria
m <- weightBin(tdm)
#converte pra matriz
transp <- t(m)
m2 <- as.matrix(transp)
write.csv(m2, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/Process_Mining_Abstracts_Binario.csv")

#A terceira representacao e a TFIDF
terms <-DocumentTermMatrix(CorpusVar,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
#converte pra matriz
m3 <- as.matrix(terms)
transp <- t(m3)
matriz <- as.matrix(transp)
write.csv(m3, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/TFIDF.csv")

