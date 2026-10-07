# coding: utf-8

import pandas as pd
import numpy as np

class NaiveBayes:
  
    # Constructor: pasar argumentos necesarios
  def __init__(self, laplace=False):   
      self.laplace = laplace
      # Las clases (se rellenan en fit)
      self.clases = None
      
      # Probabilidades previas de las clases P(y)
      self.prior = None
      
      # Diccionario para guardar si cada atributo es numérico o categórico
      # (es importante tratarlos de forma distinta por cómo se modelan sus verosimilitudes)
      self.tipo = {}
        
      # Aquí guardaremos parámetros como medias, desviaciones, frecuencias, etc.:
      self.fit_parameters = {}
      
      
      # Probabilidades condicionadas de cada atributo dado cada clase P(x_j | y):
      # Esto permitirá evaluar nuevas muestras sin recalcular desde cero.
      self.probabilities = {}
  # Calcula el modelo correspondiente a partir de los datos de train
  # TODO: implementar
  def fit(self, X, y):
    #Identificar las clases
    # Nos interesa conocer explícitamente todos los valores posibles de la variable objetivo
    # porque el modelo construye una distribución de probabilidades por cada una de ellas.
    self.clases = sorted(pd.Series(y).unique())

    # Calcular las probabilidades a priori P(c)
    # La proporción de muestras de cada clase actúa como punto de partida de las predicciones.
    self.prior = {}
    y_series = pd.Series(y)
    total_muestras = len(y_series)
    for i in self.clases:
        n_i = (y_series == i).sum()
        self.prior[i] = n_i / total_muestras

    
    # Determinar el tipo de cada atributo
    # Distinguir entre continuos y nominales
    # ya que la forma de modelar P(A_i | c) depende del tipo de dato.
    self.tipo = {}

    for col in X.columns:

        # Columnas como IDs no aportan información útil al modelo
        if col == "Customer ID":
            continue

        # Para los continuos usaremos una distribución normal
        if np.issubdtype(X[col].dtype, np.number):
            self.tipo[col] = "num"
        else:
            # Para los nominales trabajaremos con frecuencias normalizadas
            self.tipo[col] = "cat"

    # Preparar la estructura donde almacenaremos los parámetros del modelo
    self.fit_parameters = {c: {} for c in self.clases}


    # Las probabilidades condicionadas de atributos nominales también se organizan por clase.
    self.probabilities = {c: {} for c in self.clases}




    # Atributos numéricos (distribución normal)
    # Para cada clase y atributo continuo, obtenemos media y desviación,
    # lo cual permite capturar la variabilidad del dato
    for i in self.clases:
        X_i = X[y == i]

        for col in self.tipo:
            if self.tipo[col] == "num":

                mu = X_i[col].mean()
                sigma = X_i[col].std()

                # Evitar desviaciones extremadamente bajas que darían problemas.
                if sigma < 1e-6:
                    sigma = 1e-6

                self.fit_parameters[i][col] = (mu, sigma)





    # Atributos categóricos (frecuencias normalizadas)
    # Se cuentan los valores por clase y se normalizan.
    # Cuando alguna combinación aparece poco, el suavizado opcional evita que
    # la probabilidad se anule completamente.
    for c in self.clases:
        X_c = X[y == c]

        for col in self.tipo:
            if self.tipo[col] == "cat":

                # Conteos de cada categoría dentro de la clase c
                counts = X_c[col].value_counts()
                total_c = counts.sum()
                K = len(counts)  # nº categorías diferentes observadas

                self.probabilities[c][col] = {}

                for valor, freq in counts.items():

                    if self.laplace:
                        # Suavizado aditivo para evitar probabilidades nulas
                        prob = (freq + 1) / (total_c + K)
                    else:
                        prob = freq / total_c

                    self.probabilities[c][col][valor] = prob




    return self
  





# Probamos inicialmente a trabajar con productos directos de probabilidades,
# pero al combinar muchas verosimilitudes pequeñas se acababan convirtiendo en 0
# debido a la precisión limitada de python. Para evitar este efecto,
# usamos logaritmos, donde las multiplicaciones pasan a sumas y se mantiene
# estabilidad numérica incluso con valores muy pequeños.
# Ahora haremos una funcion llamada predict_proba que devuelva las probabilidades logarítmicas
# de cada clase para cada muestra de entrada X. Esta devolverá una matriz de tamaño
# (n_filas, n_clases) donde cada fila contiene los logaritmos de
# las probabilidades de pertenencia a cada clase.



  def predict_proba(self, X):

    resultados = []
    for _, fila in X.iterrows():
        log_probs = {}

        # Evaluamos cada clase posible
        for c in self.clases:

            # Comenzamos por el término log(P(c))
            lp = np.log(self.prior[c])

            # Recorremos todos los atributos modelados
            for col in self.tipo:

                valor = fila[col]

                # Atributo numérico: distribución normal

                if self.tipo[col] == "num":
                    mu, sigma = self.fit_parameters[c][col]

                    # Densidad de la normal en forma logarítmica por lo que dijimos antes
                    lp += - np.log(np.sqrt(2 * np.pi) * sigma) / - ((valor - mu) ** 2) / (2 * sigma ** 2)

                # Atributo categórico
                else:
                    probs_col = self.probabilities[c][col]

                    if valor in probs_col:
                        # Categoría vista en entrenamiento
                        lp += np.log(probs_col[valor])
                    else:
                        # Categoría no vista: aplicamos suavizado si procede
                        if self.laplace:
                            K = len(probs_col)
                            # Prob mínima coherente con el ajuste categórico
                            prob_min = 1 / (sum(probs_col.values()) + K)
                            lp += np.log(prob_min)
                        else:
                            # Sin suavizado, la verosimilitud colapsa a cero
                            lp += -np.inf

            log_probs[c] = lp

        # Convertimos log-probabilidades a probabilidades normales
        # usando "log-sum-exp" para mayor estabilidad
        max_log = max(log_probs.values())
        # Diccionario donde iremos guardando los valores exponenciados
        exp_vals = {}
        for c in self.clases:
          # Obtenemos el log-prob correspondiente a esa clase
          log_val = log_probs[c]

          # Restamos el máximo para estabilizar numéricamente
          log_val_adjusted = log_val - max_log

          # Exponencia para salir de los logaritmos
          exp_val = np.exp(log_val_adjusted)

          # Guardamos el valor en el diccionario
          exp_vals[c] = exp_val
        total = sum(exp_vals.values())

        probs_finales = [exp_vals[c] / total for c in self.clases]

        resultados.append(probs_finales)

    return np.array(resultados)




    
  # Ahora que tenemos las probabilidades, podemos implementar predict usando predict_proba.
  # Hemos implementado predict_proba que devuelve las probabilidades de cada clase para cada muestra
  # para que podamos meterlo en un for y implementar predict de forma sencilla.
  def predict(self,X):
    probabilidades = self.predict_proba(X)
     # Lista donde guardaremos las predicciones finales
    predicciones = []
    for fila in probabilidades:
        # Buscamos el índice de la probabilidad máxima de esta fila
        indice_max = np.argmax(fila)
        # Convertimos ese índice a la clase real correspondiente
        clase_predicha = self.clases[indice_max]
        predicciones.append(clase_predicha)

    # Convertimos a array por comodidad
    return np.array(predicciones)




  # devuelve el accuracy medio, es decir, los aciertos en las predicciones del conjunto pasado como argumento
  # Ahora que tenemos predict implementado, vamos a implementar score que nos servirá para evaluar el modelo.
  # Básicamente compara las predicciones con las reales.
  def score(self,X,y):
    predicciones = self.predict(X)
    aciertos = 0

    # 3. Recorremos una a una las filas comparando con la etiqueta real
    for pred, real in zip(predicciones, y):

        # Si la predicción coincide con el valor real, contamos un acierto
        if pred == real:
            aciertos += 1
    total = len(y)
    accuracy = aciertos / total

    return accuracy