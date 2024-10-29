# Inspection Data Analysis and Logistic Regression Model


Data: [data_kaggle]("https://www.kaggle.com/datasets/tjkyner/chicago-food-inspections")
## Project Overview

This project involves analyzing inspection data and building a Logistic Regression model to classify the inspection results. The dataset contains various features related to inspection records, including facility types, risks, and locations. 

### Objectives

- Perform Exploratory Data Analysis (EDA) to understand the dataset.
- Preprocess the data for modeling.
- Train a Logistic Regression model to classify inspection results.
- Evaluate the model's performance.

## Data Description

The dataset contains the following columns:

- **Inspection ID**: Unique identifier for each inspection (int64)
- **DBA Name**: Name of the establishment (object)
- **AKA Name**: Also known as name (object)
- **License #**: License number (float64)
- **Facility Type**: Type of facility (object)
- **Risk**: Risk level (object)
- **Address**: Address of the facility (object)
- **City**: City where the facility is located (object)
- **State**: State where the facility is located (object)
- **Zip**: ZIP code (float64)
- **Inspection Date**: Date of inspection (object)
- **Inspection Type**: Type of inspection (object)
- **Results**: Outcome of the inspection (object)
- **Violations**: List of violations (object)
- **Latitude**: Latitude of the facility (float64)
- **Longitude**: Longitude of the facility (float64)
- **Location**: Location data (object)

## Methodology

1. **Exploratory Data Analysis (EDA)**: 
   - Loaded the dataset and explored its structure.
   - Checked for missing values and analyzed categorical features.
   - Visualized the distribution of the target variable.

2. **Data Preprocessing**: 
   - Removed unnecessary columns and handled missing values.
   - Encoded categorical variables using Label Encoding and One-Hot Encoding.
   - Split the dataset into training and testing sets.
   - Scaled numerical features using StandardScaler.

3. **Model Training**: 
   - Trained a Logistic Regression model on the preprocessed data.

4. **Model Evaluation**: 
   - Evaluated the model using accuracy, confusion matrix, and classification report.

## Conclusion

The Logistic Regression model was successfully trained and evaluated. The accuracy and classification metrics provide insights into the model's performance on the inspection data.

---

# Análisis de Datos de Inspección y Modelo de Regresión Logística

## Descripción del Proyecto

Este proyecto implica analizar datos de inspección y construir un modelo de Regresión Logística para clasificar los resultados de la inspección. El conjunto de datos contiene varias características relacionadas con los registros de inspección, incluyendo tipos de instalaciones, riesgos y ubicaciones.

### Objetivos

- Realizar un Análisis Exploratorio de Datos (EDA) para comprender el conjunto de datos.
- Preprocesar los datos para el modelado.
- Entrenar un modelo de Regresión Logística para clasificar los resultados de la inspección.
- Evaluar el rendimiento del modelo.

## Descripción de los Datos

El conjunto de datos contiene las siguientes columnas:

- **Inspection ID**: Identificador único para cada inspección (int64)
- **DBA Name**: Nombre del establecimiento (objeto)
- **AKA Name**: Nombre alternativo (objeto)
- **License #**: Número de licencia (float64)
- **Facility Type**: Tipo de instalación (objeto)
- **Risk**: Nivel de riesgo (objeto)
- **Address**: Dirección de la instalación (objeto)
- **City**: Ciudad donde se encuentra la instalación (objeto)
- **State**: Estado donde se encuentra la instalación (objeto)
- **Zip**: Código postal (float64)
- **Inspection Date**: Fecha de la inspección (objeto)
- **Inspection Type**: Tipo de inspección (objeto)
- **Results**: Resultado de la inspección (objeto)
- **Violations**: Lista de violaciones (objeto)
- **Latitude**: Latitud de la instalación (float64)
- **Longitude**: Longitud de la instalación (float64)
- **Location**: Datos de ubicación (objeto)

## Metodología

1. **Análisis Exploratorio de Datos (EDA)**:
   - Se cargó el conjunto de datos y se exploró su estructura.
   - Se verificaron los valores faltantes y se analizaron las características categóricas.
   - Se visualizó la distribución de la variable objetivo.

2. **Preprocesamiento de Datos**:
   - Se eliminaron columnas innecesarias y se manejaron los valores faltantes.
   - Se codificaron variables categóricas utilizando codificación de etiquetas y codificación one-hot.
   - Se dividió el conjunto de datos en conjuntos de entrenamiento y prueba.
   - Se escalaron características numéricas utilizando StandardScaler.

3. **Entrenamiento del Modelo**:
   - Se entrenó un modelo de Regresión Logística con los datos preprocesados.

4. **Evaluación del Modelo**:
   - Se evaluó el modelo utilizando precisión, matriz de confusión e informe de clasificación.

## Conclusión

El modelo de Regresión Logística fue entrenado y evaluado con éxito. La precisión y las métricas de clasificación proporcionan información sobre el rendimiento del modelo en los datos de inspección.

# Informe de Clasificación

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.80   | 0.82     | 11783   |
| 1     | 0.57      | 0.08   | 0.15     | 153     |
| 2     | 0.00      | 0.00   | 0.00     | 16      |
| 3     | 0.00      | 0.00   | 0.00     | 10      |
| 4     | 0.96      | 0.99   | 0.97     | 26526   |
| 5     | 0.80      | 0.80   | 0.80     | 9970    |

### Métricas Generales

- **Exactitud (Accuracy)**: 0.90 (90% de las predicciones fueron correctas en general)
- **Macro Promedio (Macro Avg)**:
  - Precisión: 0.53
  - Recall: 0.44
  - F1-Score: 0.46
- **Promedio Ponderado (Weighted Avg)**:
  - Precisión: 0.89
  - Recall: 0.90
  - F1-Score: 0.90

### Interpretación de las Métricas

1. **Clase 0**: La precisión es alta (0.85) con buen recall (0.80), lo cual da un F1-score de 0.82, indicando un buen desempeño en esta clase.
   
2. **Clase 1**: Aunque tiene una precisión de 0.57, el recall es bajo (0.08), resultando en un F1-score bajo (0.15), lo que indica que el modelo no está capturando bien los ejemplos de esta clase.

3. **Clase 2 y Clase 3**: Tienen precisión, recall y F1-score de 0.00 debido a que no hay predicciones hechas en estas clases, como indica el mensaje de advertencia en el reporte.

4. **Clase 4**: Muestra un alto rendimiento, con precisión de 0.96, recall de 0.99 y F1-score de 0.97. Es la clase mejor clasificada.

5. **Clase 5**: Tiene un rendimiento adecuado con un F1-score de 0.80, gracias a su precisión y recall ambos en 0.80.

### Para cargar el modelo desde json

Para cargar el modelo desde el json puedes usar el siguiente codigo:

```py
from sklearn.linear_model import LogisticRegression
import json
import numpy as np

# Función para cargar el modelo desde JSON
def load_model_from_json(file_path):
    with open(file_path, 'r') as json_file:
        model_data = json.load(json_file)

    # Crear una instancia del modelo de regresión logística
    model = LogisticRegression()

    # Asignar los coeficientes e intercepto al modelo
    model.coef_ = np.array(model_data['coeficients'])
    model.intercept_ = np.array(model_data['intercept'])
    model.classes_ = np.array(model_data['classes'])

    return model

# Ejemplo de uso
loaded_model = load_model_from_json('../model/modelo.json')


```
