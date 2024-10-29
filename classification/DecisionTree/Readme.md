## K-Nearest Neighbors (KNN)

### English

**What is K-Nearest Neighbors (KNN)?**

K-Nearest Neighbors (KNN) is a simple, non-parametric, and instance-based learning algorithm used for both classification and regression tasks. It works by identifying the "k" nearest data points (or "neighbors") to a target point within the feature space. The algorithm classifies the target point based on the majority class of its neighbors (for classification) or averages their values (for regression).

**When is KNN Used?**

KNN is widely used when:
1. The data is relatively small and low-dimensional, as KNN can be computationally expensive for large datasets.
2. Simplicity and interpretability are desired, as KNN provides easy-to-understand results.
3. An application requires a "local" analysis, where the model focuses on neighbors' behavior rather than attempting to fit a global trend.

### Español

**¿Qué es K-Nearest Neighbors (KNN)?**

K-Nearest Neighbors (KNN) es un algoritmo de aprendizaje simple, no paramétrico y basado en instancias, que se utiliza tanto para tareas de clasificación como de regresión. Funciona identificando los "k" puntos de datos más cercanos (o "vecinos") a un punto objetivo dentro del espacio de características. El algoritmo clasifica el punto objetivo en función de la clase mayoritaria de sus vecinos (para clasificación) o promedia sus valores (para regresión).

**¿Cuándo se usa KNN?**

KNN es ampliamente utilizado cuando:
1. Los datos son relativamente pequeños y de baja dimensionalidad, ya que KNN puede ser costoso computacionalmente para grandes conjuntos de datos.
2. Se desea simplicidad e interpretabilidad, ya que KNN proporciona resultados fáciles de entender.
3. Una aplicación requiere un análisis "local", donde el modelo se enfoca en el comportamiento de los vecinos en lugar de intentar ajustar una tendencia global.


# Academic Dropout Prediction Using Label Propagation and SMOTEENN

## Description (Descripción)

This project explores the prediction of academic dropout among engineering students using data analysis, Label Propagation, and class balancing techniques. The dataset undergoes preprocessing, feature reduction, and visualization of model performance to gain insights into patterns that contribute to student dropout.

Este proyecto explora la predicción de la deserción académica entre estudiantes de ingeniería mediante análisis de datos, propagación de etiquetas y técnicas de balanceo de clases. Los datos son preprocesados, se realiza reducción de variables y se visualiza el rendimiento del modelo para entender patrones que contribuyen a la deserción estudiantil.

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

