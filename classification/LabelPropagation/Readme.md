## Label Propagation with K-Nearest Neighbors (KNN)

### English

**What is Label Propagation?**

Label Propagation is a semi-supervised learning algorithm that leverages both labeled and unlabeled data for classification tasks. It operates by spreading labels through the dataset based on the similarities among data points. The algorithm assigns labels to unlabeled data points by propagating the labels of the nearest neighbors.

In the context of Label Propagation, the K-Nearest Neighbors (KNN) algorithm is used as a kernel to measure the similarity between data points. This means that when Label Propagation is applied with a KNN kernel, it considers the "k" nearest neighbors to each data point to decide how to propagate the labels.

**When is Label Propagation with KNN Used?**

Label Propagation with KNN is particularly useful when:
1. You have a limited amount of labeled data and a larger pool of unlabeled data.
2. The relationships between data points are non-linear and can be better captured using local information from neighbors.
3. You want to improve classification performance in scenarios where the dataset is imbalanced or contains noisy data.

### Español

**¿Qué es la Propagación de Etiquetas?**

La Propagación de Etiquetas es un algoritmo de aprendizaje semi-supervisado que aprovecha tanto datos etiquetados como no etiquetados para tareas de clasificación. Funciona propagando etiquetas a través del conjunto de datos basándose en las similitudes entre los puntos de datos. El algoritmo asigna etiquetas a los puntos de datos no etiquetados propagando las etiquetas de los vecinos más cercanos.

En el contexto de la Propagación de Etiquetas, el algoritmo K-Nearest Neighbors (KNN) se utiliza como un kernel para medir la similitud entre los puntos de datos. Esto significa que cuando se aplica la Propagación de Etiquetas con un kernel KNN, se consideran los "k" vecinos más cercanos a cada punto de datos para decidir cómo propagar las etiquetas.

**¿Cuándo se usa la Propagación de Etiquetas con KNN?**

La Propagación de Etiquetas con KNN es particularmente útil cuando:
1. Se tiene una cantidad limitada de datos etiquetados y un conjunto más grande de datos no etiquetados.
2. Las relaciones entre los puntos de datos son no lineales y pueden ser mejor capturadas utilizando información local de los vecinos.
3. Se desea mejorar el rendimiento de clasificación en situaciones donde el conjunto de datos está desbalanceado o contiene datos ruidosos.



# Academic Dropout Prediction Using Label Propagation with KNN

Description (Descripción)

This project explores the prediction of academic dropout among engineering students using data analysis, Label Propagation with a KNN kernel, and class balancing techniques. The dataset undergoes preprocessing, feature reduction, and visualization of model performance to gain insights into patterns that contribute to student dropout.

Este proyecto explora la predicción de la deserción académica entre estudiantes de ingeniería mediante análisis de datos, Propagación de Etiquetas con un kernel KNN y técnicas de balanceo de clases. Los datos son preprocesados, se realiza reducción de variables y se visualiza el rendimiento del modelo para entender patrones que contribuyen a la deserción estudiantil.

# Prerequisites (Requisitos)

Install the following libraries: Instale las siguientes bibliotecas:

```py

pip install pandas matplotlib seaborn scikit-learn imbalanced-learn

```

# Steps (Pasos)

1. Load Data
    Cargar Datos
    * Load and explore the dataset.
    * Cargar y explorar el conjunto de datos.

2. Data Preprocessing
    Preprocesamiento de Datos
    * Remove redundant and irrelevant features.
    * Eliminar características redundantes e irrelevantes.
3. Data Visualization
    Visualización de Datos
    * Plot dropout distribution and correlation matrix.
    * Graficar la distribución de la deserción y la matriz de correlación.

4. Encode Target Variable
    Codificar Variable Objetivo
    * Encode the target variable "DESERSIÓN" for modeling.
    * Codificar la variable objetivo "DESERSIÓN" para el modelado.

```py
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['DESERSIÓN'] = labelencoder.fit_transform(data['DESERSIÓN'])
```

5. Data Balancing with SMOTEENN
    Balanceo de Datos con SMOTEENN
    * Balance classes in the training data using SMOTEENN.
    * Balancear las clases en los datos de entrenamiento usando SMOTEENN.

6. Model Training and Evaluation
    Entrenamiento y Evaluación del Modelo
    * Train and evaluate the Label Propagation model, assessing its performance.
    * Entrenar y evaluar el modelo de Propagación de Etiquetas, evaluando su rendimiento.
7. Performance Metrics and Visualization
    Métricas de Rendimiento y Visualización
    * Use accuracy, classification report, and confusion matrix.
    * Utilizar precisión, informe de clasificación y matriz de confusión.
8. Dimensionality Reduction for Visualization
    Reducción de Dimensionalidad para Visualización
    * Use PCA to reduce data to two dimensions for plotting.
    * Usar PCA para reducir los datos a dos dimensiones y graficarlos.

# Output (Salida)
* Accuracy and Classification Report for the model's performance.
* Confusion Matrix to visualize true vs. predicted values.
* 2D Visualization of predicted classes.

