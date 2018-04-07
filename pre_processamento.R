#Representações por frequência, binária e tfidf

#é necessário rodar os comandos abaixo antes de continuar, para que as bibliotecas usadas sejam instaladas (se ainda não estiverem instaladas)
#install.packages('tm')
#install.packages('SnowballC')
#install.packages('stringr')

#biblioteca usada especificamente para text-mining
library(tm)
#biblioteca que disponibiliza o uso do stemming (redução de palavras para o seu radical)
library(SnowballC)
#biblioteca que disponibiliza funções para manipulação de strings
library(stringr)

#captura o diretório
#Nesse caso, para rodar, é necessário alterar o diretório para o local onde estão os textos no seu computador
#Pega todos os textos que estão nesse diretório e transforma num corpus
CorpusVar <- Corpus(DirSource("C:/Users/Mariana/Desktop/IA/teste_R"))

#deixa todas as letras em minúsculo
CorpusVar <- tm_map(CorpusVar, content_transformer(tolower))

#remove pontuação dos textos
CorpusVar <- tm_map(CorpusVar, removePunctuation)

#remove os números dos textos
CorpusVar <- tm_map(CorpusVar, removeNumbers)

#retira as stop words (língua inglesa)
CorpusVar <- tm_map(CorpusVar, removeWords, stopwords("english"))

#remove espaços em branco que são desnecessários
CorpusVar <- tm_map(CorpusVar, stripWhitespace)

#reduz as palavras para o seu radical (língua inglesa)
CorpusVar <- tm_map(CorpusVar, stemDocument, language="english")

#A primeira representação é a por frequência
#transforma o corpus em uma matriz de frequência
tdm <- TermDocumentMatrix(CorpusVar)
#converte pra matriz
m <- as.matrix(tdm)
#colocar pasta específica de saída com o nome do arquivo
write.csv(m, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/Process_Mining_Abstracts_TF.csv")

#A segunda representação é a binária
m <- weightBin(tdm)
#converte pra matriz
m2 <- as.matrix(m)
#colocar pasta específica de saída com o nome do arquivo
write.csv(m2, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/Process_Mining_Abstracts_Binario.csv")

#A terceira representação é a TFIDF
terms <-DocumentTermMatrix(CorpusVar,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
#converte pra matriz
m3 <- as.matrix(terms)
#colocar pasta específica de saída com o nome do arquivo                           
write.csv(m3, file="C:/Users/Mariana/Desktop/IA/Saidas_teste/TFIDF.csv")

