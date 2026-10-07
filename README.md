# PRÁCTICAS ML 1

Prácticas de la asignatura **Aprendizaje Automático I (ML 1)** (2º curso, 1er cuatrimestre) del **Grado en Ciencia e Ingeniería de Datos**.

Cuatro prácticas en Jupyter Notebook que recorren las fases de un proyecto de machine learning clásico: preprocesado, regresión, clasificación y reducción de dimensionalidad. Varios modelos se implementan también "a mano" (predicción lineal, curva ROC, Naïve Bayes) y se comparan con los de scikit-learn para comprobar que dan el mismo resultado.

## Contenido

| Práctica | Tema | Dataset | Técnicas principales |
|---|---|---|---|
| [PRÁCTICA 1](PRÁCTICA1/) | Preprocesado de datos | House Prices (1460 viviendas, 81 atributos) | Análisis exploratorio, tipos de variables, tratamiento de nulos, codificación de categóricas (ordinales y one-hot) |
| [PRÁCTICA 2](PRÁCTICA2/) | Regresión | House Prices preprocesado · Asistencia a partidos de béisbol (MLB) | Eliminación de colinealidad, regresión lineal (sklearn y statsmodels), predicción manual, análisis de residuos, transformación logarítmica del *target*, p-valores, Lasso y Ridge |
| [PRÁCTICA 3](PRÁCTICA3/) | Clasificación | Fuga de clientes de una telefónica (*churn*) | Limpieza y *outliers* (z-score), modelo *dummy*, regresión logística, matriz de confusión, ROC/AUC manual, F1, curva Precision-Recall, umbral óptimo por costes, Naïve Bayes implementado desde cero |
| [PRÁCTICA 4](PRÁCTICA4/) | Reducción de dimensionalidad | Fuga de clientes (reutiliza el preprocesado de la P3) | Estandarización, PCA, varianza explicada, incorrelación de componentes |

### Algunos resultados

- Regresión (P2): la regresión lineal sobre House Prices alcanza R² = 0.94 (train) / 0.88 (test); al aplicar logaritmo al precio sube a 0.97 / 0.93. Los resultados de sklearn y statsmodels coinciden.
- Regresión logística (P3): AUC = 0.729, idéntico entre la implementación manual y sklearn. Ajustando el umbral según el coste de los errores, el mínimo aparece en torno a 0.064.
- Naïve Bayes (P3): la implementación propia (gaussiana para numéricas + frecuencias para categóricas) obtiene ~0.66 de *accuracy*, frente a ~0.68 de `CategoricalNB`/`MultinomialNB` y ~0.63 de `GaussianNB`.

## Estructura

```
.
├── PRÁCTICA1/   # Preprocesado: AA1_P1_.ipynb + ejemplo de categóricas no ordenadas
├── PRÁCTICA2/   # Regresión: semanas 1-2 (housing) y semana 3 (béisbol)
├── PRÁCTICA3/   # Clasificación: semanas 1-3, plantilla y clase Naïve Bayes
└── PRÁCTICA4/   # PCA
```

Cada carpeta incluye el enunciado en PDF, los datasets en CSV y los notebooks resueltos. Los ficheros `predicciones.csv` son las predicciones entregadas sobre los conjuntos de explotación.

## Cómo ejecutarlo

```bash
git clone https://github.com/Code-Cram/PRACTICAS-ML-1.git
cd PRACTICAS-ML-1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

> Los notebooks leen los CSV con rutas relativas, así que hay que ejecutarlos desde su propia carpeta y en orden, de arriba abajo: el dataframe se va modificando a lo largo del notebook.

## Tecnologías

Python · NumPy · pandas · Matplotlib · seaborn · SciPy · scikit-learn · statsmodels · Jupyter

## Autoría

- Marc Martínez Arias
- Pedro Barros Bobadilla (colaborador)

Los enunciados y los datasets son del profesorado de la asignatura y se incluyen solo como referencia.
