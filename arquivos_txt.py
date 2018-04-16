# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:45:35 2018

@author: Mariana
"""

#esse código quebra os documentos do txt "Process Mining Abstracts" em vários txt diferentes
#Quebrar documentos do arquivo de texto em vários arquivos txt

import pandas as pd
import numpy as np

#funcao que carrega o arquivo de entrada para kmeans
def carregaTXT(nome):    
    return np.loadtxt(nome)

##################################################################################################################################

arquivo = 'C:/Users/Mariana/Desktop/IA/Textos/Entrada1/ProcessMiningAbstracts.txt'

f = open(arquivo)
data = f.readlines()
f.close()

i = 0

while i < len(data):
    
    if len(data[i]) == 1:
        
        del data[i]
    
    i += 1

doc = 0

data.append('Fim')

while doc < len(data):
    arquivo = open('C:/Users/Mariana/Desktop/IA/Textos/Entrada1/Textos/Doc'+str(doc)+'.txt', 'w')
    arquivo.write(data[doc])
    doc += 1







