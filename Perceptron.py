#Importando a biblioteca numpy
import numpy as np

#Criando a nossa base de dados
base = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

#Inicializando os pesos de forma aleatória.
pesos_camada1 = 2*np.random.random((2, 1))-1
peso_bias = 2*np.random.random((1, 1)) -1

#Função sigmoide e sua derivada
def funSigm(x, dev=False):
    if dev == False:
        return 1 / (1 + np.exp(-x))
    return x * (1 - x)

#Número de vezes que treinaremos a rede.
epocas = 10000

for epoca in range(epocas):
    
    #Variável para tirar a média dos erros
    media = 0
    
    #Embaralhando os nossos dados
    np.random.shuffle(base)
    
    #Dividindo entre entradas e saidas
    entradas = base[:, :2]
    saidas = base[:, 2]
    
    #For para percorrer nossa matriz de entradas
    for pos, entrada in enumerate(entradas):
        
        #Calculos do "impulso nervoso"
        Result_camada1 = funSigm(np.dot(entrada, pesos_camada1) + peso_bias)
        
        #O quão longe estamos de um bom resultado ?
        erro = Result_camada1 - saidas[pos]
        
        media += erro
        
        #Calculo do Delta
        Delta = 2 * erro *  funSigm(Result_camada1, dev=True)
        
        #Backpropagation 
        pesos_camada1 -= Delta * entrada.reshape(2, 1) * 0.5
        peso_bias -= Delta * 0.5
        
    if epoca % 1000 == 0:
        print("Erro: {}".format(media/4))


