# Understanding Support Vector Machines (SVM)

This document provides an overview of Support Vector Machines (SVM), a powerful supervised learning algorithm primarily used for classification tasks. We will explore how SVM works and visualize key concepts with graphs.

## Table of Contents

- [Introduction to SVM](#introduction-to-svm)
- [How SVM Works](#how-svm-works)
- [Key Concepts](#key-concepts)
- [Visualization of SVM Concepts](#visualization-of-svm-concepts)
- [Conclusion](#conclusion)

## Introduction to SVM

Support Vector Machines are supervised learning algorithms that analyze data for classification and regression analysis. The main idea is to find the best separating hyperplane that divides the classes in the feature space.

## How SVM Works

1. **Data Representation**: Each data point is represented as a vector in a high-dimensional space. For example, in a two-dimensional space, data points are plotted on a 2D plane.

2. **Hyperplane**: SVM attempts to find a hyperplane that separates the classes. In a two-dimensional space, this hyperplane is a line.

3. **Maximizing the Margin**: SVM aims to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class (support vectors).

4. **Kernel Trick**: For non-linearly separable data, SVM uses kernel functions to project the data into a higher-dimensional space where a linear separation is possible.

## Key Concepts

- **Support Vectors**: The data points that are closest to the hyperplane and influence its position and orientation.
- **Margin**: The distance between the hyperplane and the nearest support vectors from either class.
- **Kernel Functions**: Functions that allow SVM to operate in high-dimensional spaces without explicitly computing the coordinates of the data in that space.

## Visualization of SVM Concepts

### 1. Linearly Separable Data

![Linearly Separable Data](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)

This graph shows two classes that can be perfectly separated by a straight line (hyperplane). The support vectors are highlighted, and the margin is illustrated.

### 2. Non-Linearly Separable Data with RBF Kernel

![Non-Linearly Separable Data](https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/05/support-vector-machine-svm.jpg?fit=1200%2C675&ssl=1&resize=1280%2C720)

In this case, the data cannot be separated by a straight line. The SVM uses an RBF kernel to project the data into a higher-dimensional space, allowing a linear separation.

### 3. Margin and Support Vectors

![Margin and Support Vectors](https://media.geeksforgeeks.org/wp-content/uploads/20240226144048/image-217.webp)

This diagram illustrates the margin between the hyperplane and the support vectors. The goal of SVM is to maximize this margin.

## Conclusion

Support Vector Machines are a robust and versatile classification technique that can handle both linearly and non-linearly separable data. Understanding the concepts of hyperplanes, margins, and support vectors is crucial for effectively applying SVMs in various machine learning tasks.


# README - Diabetes Classification using SVM

This document details the process followed to classify the diabetes dataset using a Support Vector Machine (SVM) model. An exploratory data analysis (EDA) was conducted, the model was optimized through hyperparameter tuning, and the class separation was visualized.

## Content

- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [SVM Model](#svm-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results Visualization](#results-visualization)
- [Saving the Model](#saving-the-model)
- [Conclusion](#conclusion)

## Dataset Description

The dataset used is the diabetes dataset, available in the `scikit-learn` library. It contains data on various characteristics of patients that can be used to predict whether a patient has diabetes. This dataset includes the following columns:

- **Age**: Age of the patient.
- **Sex**: Sex of the patient (0: female, 1: male).
- **Body Mass Index (BMI)**: Ratio of weight to height.
- **Blood Pressure**: Patient's blood pressure.
- **Glucose**: Blood glucose levels.
- **Insulin**: Blood insulin levels.
- **Insulin Function**: Measure of insulin sensitivity.
- **Glucose Function**: Measure of glucose response.

## Exploratory Data Analysis (EDA)

An exploratory analysis was performed to better understand the dataset. This includes:

- Visualization of the first rows of the dataset.
- Descriptive statistics showing the mean, standard deviation, minimum and maximum values.
- Information about data types and the presence of null values.
- Creation of a pair plot to observe the distributions of features and their relationship with the target variable.

## SVM Model

Support Vector Machine (SVM) is a supervised learning algorithm primarily used for classification, although it can also be applied to regression problems. SVM seeks a hyperplane in a high-dimensional space that optimally separates different classes of data. This hyperplane maximizes the distance between classes, known as the margin.

### How SVM Works:

- **Class Separation**: SVM attempts to find a hyperplane that separates different classes of data in a feature space. For linearly separable data, this hyperplane perfectly divides the classes.

- **Maximum Margin**: The main idea is to maximize the margin, which is the distance between the hyperplane and the nearest samples of any class (called support vectors).

- **Kernel Trick**: When the data is not linearly separable, SVM uses kernel functions to project the data into a higher-dimensional space where they can be linearly separated. Types of kernels include linear, polynomial, and radial (RBF).

## Hyperparameter Tuning

`GridSearchCV` was used to perform an exhaustive search for the best hyperparameters for the SVM model. This includes:

- **C**: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error.
- **gamma**: Kernel parameter that defines the influence of a single training point.
- **kernel**: Type of kernel used (linear, RBF, polynomial).
- **degree**: Degree of the polynomial if a polynomial kernel is used.

## Model Evaluation

The model was evaluated using metrics such as accuracy, confusion matrix, and classification report, both for the training set and the test set.

## Results Visualization

A visualization of the class separation in the test set was generated using Seaborn.

## Conclusion

An SVM model was developed to classify diabetes data. Through normalization, hyperparameter tuning, and results visualization, a robust model was created. This process is fundamental to understanding the importance of tuning and evaluation in machine learning models.

For more information on SVM and its applications, you can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/svm.html).


# README - Clasificación de Diabetes usando SVM

Este documento detalla el proceso seguido para clasificar el dataset de diabetes utilizando un modelo de Máquina de Vectores de Soporte (SVM). Se llevó a cabo un análisis exploratorio de datos (EDA), se optimizó el modelo mediante búsqueda de hiperparámetros y se visualizó la separación de las clases.

## Contenido

- [Descripción del Dataset](#descripción-del-dataset)
- [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
- [Modelo SVM](#modelo-svm)
- [Ajuste de Hiperparámetros](#ajuste-de-hiperparámetros)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Visualización de Resultados](#visualización-de-resultados)
- [Guardar el Modelo](#guardar-el-modelo)
- [Conclusión](#conclusión)

## Descripción del Dataset

El dataset utilizado es el de diabetes, disponible en la biblioteca `scikit-learn`. Contiene datos sobre varias características de los pacientes que se pueden usar para predecir si un paciente tiene diabetes. Este dataset tiene las siguientes columnas:

- **Edad**: Edad del paciente.
- **Sexo**: Sexo del paciente (0: femenino, 1: masculino).
- **Índice de Masa Corporal (BMI)**: Relación entre el peso y la altura.
- **Presión Sanguínea**: Presión arterial del paciente.
- **Glucosa**: Niveles de glucosa en sangre.
- **Insulina**: Niveles de insulina en sangre.
- **Función de la Insulina**: Medida de la sensibilidad a la insulina.
- **Función de la Glucosa**: Medida de la respuesta a la glucosa.

## Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exploratorio para comprender mejor el dataset. Esto incluye:

- Visualización de las primeras filas del dataset.
- Estadísticas descriptivas que muestran la media, desviación estándar, valores mínimos y máximos.
- Información sobre los tipos de datos y la presencia de valores nulos.
- Creación de un gráfico de pares para observar las distribuciones de las características y su relación con la variable objetivo.

## Modelo SVM

La Máquina de Vectores de Soporte (SVM) es un algoritmo de aprendizaje supervisado que se utiliza principalmente para clasificación, aunque también puede aplicarse a problemas de regresión. La SVM busca un hiperplano en un espacio de alta dimensión que separe las diferentes clases de datos de manera óptima. Este hiperplano maximiza la distancia entre las clases, lo que se conoce como el margen.

### Cómo Funciona SVM:

- **Separación de Clases**: SVM intenta encontrar un hiperplano que separe las diferentes clases de datos en un espacio de características. Para datos linealmente separables, este hiperplano divide perfectamente las clases.

- **Margen Máximo**: La idea principal es maximizar el margen, que es la distancia entre el hiperplano y las muestras más cercanas de cualquier clase (llamadas vectores de soporte).

- **Kernel Trick**: Cuando los datos no son linealmente separables, SVM utiliza funciones kernel para proyectar los datos en un espacio de mayor dimensión donde sí pueden ser separados linealmente. Los tipos de kernel incluyen lineal, polinómico y radial (RBF).

## Ajuste de Hiperparámetros

Se utilizó `GridSearchCV` para realizar una búsqueda exhaustiva de los mejores hiperparámetros para el modelo SVM. Esto incluye:

- **C**: Parámetro de regularización que controla el compromiso entre maximizar el margen y minimizar el error de clasificación.
- **gamma**: Parámetro del kernel que define la influencia de un solo punto de entrenamiento.
- **kernel**: Tipo de kernel utilizado (lineal, RBF, polinómico).
- **degree**: Grado del polinomio si se usa un kernel polinómico.

# Evaluación del Modelo

Se evaluó el modelo utilizando métricas como la precisión, la matriz de confusión y el informe de clasificación, tanto para el conjunto de entrenamiento como para el conjunto de prueba.

# Visualización de Resultados

Se generó una visualización de la separación de clases en el conjunto de prueba utilizando seaborn.

## Conclusión

Se desarrolló un modelo de SVM para clasificar los datos de diabetes. A través de la normalización, la búsqueda de hiperparámetros y la visualización de resultados, se logró crear un modelo robusto. Este proceso es fundamental para entender la importancia del ajuste y la evaluación en modelos de aprendizaje automático.

Para más información sobre SVM y sus aplicaciones, puedes consultar la [documentación de scikit-learn](https://scikit-learn.org/stable/modules/svm.html).
