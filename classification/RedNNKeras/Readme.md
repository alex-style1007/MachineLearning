<h1>What is a Neural Network?</h1>
<h1>¿Qué es una Red Neuronal?</h1>

<p>A neural network is a computational model inspired by the way biological neural networks in the human brain work. It consists of interconnected nodes (neurons) that process input data to make predictions or classifications. Neural networks can learn from data, adjust their parameters, and improve their accuracy over time.</p>
<p>Una red neuronal es un modelo computacional inspirado en la forma en que funcionan las redes neuronales biológicas en el cerebro humano. Consiste en nodos interconectados (neuronas) que procesan datos de entrada para hacer predicciones o clasificaciones. Las redes neuronales pueden aprender de los datos, ajustar sus parámetros y mejorar su precisión con el tiempo.</p>

<h2>How Neural Networks Work</h2>
<h2>¿Cómo Funcionan las Redes Neuronales?</h2>

<p>Neural networks typically consist of an input layer, one or more hidden layers, and an output layer. Each layer consists of neurons that apply a weighted sum followed by an activation function to transform the input data. This process allows the network to learn complex patterns in the data.</p>
<p>Las redes neuronales típicamente consisten en una capa de entrada, una o más capas ocultas y una capa de salida. Cada capa consiste en neuronas que aplican una suma ponderada seguida de una función de activación para transformar los datos de entrada. Este proceso permite a la red aprender patrones complejos en los datos.</p>

<h2>Architecture of Convolutional Neural Networks (CNN)</h2>
<h2>Arquitectura de Redes Neuronales Convolucionales (CNN)</h2>

<p>In the provided code, a Convolutional Neural Network (CNN) is used, which is particularly effective for image classification tasks. CNNs consist of convolutional layers that automatically learn spatial hierarchies of features from images. They use filters (kernels) to extract important features, followed by pooling layers to reduce dimensionality.</p>
<p>En el código proporcionado, se utiliza una Red Neuronal Convolucional (CNN), que es particularmente efectiva para tareas de clasificación de imágenes. Las CNN consisten en capas convolucionales que aprenden automáticamente jerarquías espaciales de características a partir de imágenes. Utilizan filtros (kernels) para extraer características importantes, seguidos de capas de agrupamiento para reducir la dimensionalidad.</p>



<h2>Applications of Neural Networks</h2>
<h2>Aplicaciones de las Redes Neuronales</h2>

<p>Neural networks have a wide range of applications, including:</p>
<p>Las redes neuronales tienen una amplia gama de aplicaciones, incluyendo:</p>

<ul>
    <li>Image and video recognition</li>
    <li>Procesamiento de imágenes y videos</li>
    <li>Natural language processing</li>
    <li>Procesamiento del lenguaje natural</li>
    <li>Autonomous vehicles</li>
    <li>Vehículos autónomos</li>
    <li>Medical diagnosis</li>
    <li>Diagnóstico médico</li>
</ul>

<h2>Conclusion</h2>
<h2>Conclusión</h2>

<p>Neural networks, especially CNNs, are powerful tools for tasks involving image data. They enable machines to learn from examples and make predictions based on complex patterns.</p>
<p>Las redes neuronales, especialmente las CNN, son herramientas poderosas para tareas que involucran datos de imágenes. Permiten a las máquinas aprender de ejemplos y hacer predicciones basadas en patrones complejos.</p>


<h1>Documentation of the Code: Handwritten Digit Classification with PyTorch</h1>
<h1>Documentación del Código: Clasificación de Dígitos Manuscritos con PyTorch</h1>

<p>This script uses PyTorch and the MNIST dataset to build and train a convolutional neural network (CNN) model that classifies handwritten digits.</p>
<p>Este script utiliza PyTorch y la base de datos MNIST para construir y entrenar un modelo de red neuronal convolucional (CNN) que clasifica dígitos manuscritos.</p>

<h2>Dependencies</h2>
<h2>Dependencias</h2>

<p>Make sure you have the following libraries installed:</p>
<p>Asegúrate de tener instaladas las siguientes bibliotecas:</p>

<pre><code>pip install torch torchvision matplotlib seaborn scikit-learn</code></pre>

<h2>Library Imports</h2>
<h2>Importación de Librerías</h2>

<pre><code>import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix</code></pre>

<h2>Data Loading</h2>
<h2>Carga de Datos</h2>

<p>The MNIST dataset is loaded and a random image is displayed.</p>
<p>Se carga el conjunto de datos MNIST y se muestra una imagen aleatoria.</p>

<pre><code>data = MNIST(root='/content/', download=True)  # Load the MNIST dataset
print(data.data.shape)  # Dimensions of the images
print(data.targets.shape)  # Dimensions of the labels

ind = np.random.randint(0, 60000)  # Random index
img = data.data[ind].numpy()  # Corresponding image to the index
plt.figure(figsize=(3, 3))
plt.imshow(img)  # Show image
plt.title(str(data.targets[ind]))  # Show label
plt.axis('off')  
plt.show()</code></pre>

<h2>Device Setup</h2>
<h2>Configuración de Dispositivo</h2>

<p>It determines whether to use GPU or CPU for training.</p>
<p>Se determina si se utilizará GPU o CPU para el entrenamiento.</p>

<pre><code>device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)  # Show the device used

# Transfer data to the device
X = data.data.to(device)
Y = data.targets.to(device)

# Show the shapes of the tensors
print("Shape of X:", X.shape)
print("Y shape:", Y.shape)</code></pre>

<h2>Data Preparation</h2>
<h2>Preparación de Datos</h2>

<p>A list of tuples is created for the data, adding an extra dimension for each input tensor.</p>
<p>Se crea una lista de tuplas para los datos, añadiendo una dimensión extra para cada tensor de entrada.</p>

<pre><code># Create list of tuples
data = [(torch.unsqueeze(xi, 0), yi) for xi, yi in zip(X, Y)]

# Access the first tag in the list
first_tag = data[0][1]
print("First tag:", first_tag)  # Show first label</code></pre>

<h2>DataLoader</h2>
<h2>DataLoader</h2>

<p>A <code>DataLoader</code> is set up to handle batches of data during training.</p>
<p>Se configura un <code>DataLoader</code> para manejar el batch de datos durante el entrenamiento.</p>

<pre><code># Dataloader
data_loader = DataLoader(data, batch_size=45000, shuffle=True)</code></pre>

<h2>Neural Network Model</h2>
<h2>Modelo de Red Neuronal</h2>

<p>The architecture of the neural network is defined.</p>
<p>Se define la arquitectura de la red neuronal.</p>

<pre><code># Model
model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=6, padding=1, kernel_size=5),  # Convolutional layer
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
    torch.nn.Conv2d(in_channels=6, out_channels=16, padding=1, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(120, 84),
    torch.nn.ReLU(),
    torch.nn.Linear(84, 10),
    torch.nn.Dropout(p=0.5),
    torch.nn.LogSoftmax(dim=-1)
).to(device)  # Send model to device</code></pre>

<h2>Training Configuration</h2>
<h2>Configuración del Entrenamiento</h2>

<p>The learning rate, loss function, and optimizer are defined.</p>
<p>Se definen la tasa de aprendizaje, la función de pérdida y el optimizador.</p>

<pre><code>learning_rate = 1e-5
loss_fn = torch.nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)  # Optimizer</code></pre>

<h2>Model Training</h2>
<h2>Entrenamiento del Modelo</h2>

<p>The model is trained over multiple epochs, and the loss is recorded at each step.</p>
<p>El modelo se entrena a través de múltiples épocas, y se registra la pérdida en cada paso.</p>

<pre><code>errores = []
for epoch in range(400):
    for xi, yi in data_loader:
        optimizer.zero_grad()  # Reset gradients
        y_pred = model(xi.float())  # Make predictions
        loss = loss_fn(y_pred, yi)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        errores.append(loss.cpu().detach().item())  # Store loss
    if epoch % 20 == 0:
        print('Epoch: ', epoch, ', Loss: ', loss.item())

plt.plot(errores)  # Plot errors
plt.show()</code></pre>

<h2>Model Evaluation</h2>
<h2>Evaluación del Modelo</h2>

<p>The test dataset is loaded and predictions are made.</p>
<p>Se carga el conjunto de datos de prueba y se hacen predicciones.</p>

<pre><code>data = MNIST(root='/content/', download=True, train=False)
X = data.data.float().to(device)
Y = data.targets.long().to(device)
print(X.shape)
print(Y.shape)

# Make predictions
y_pred = model(X.unsqueeze(1))  # Add dimension for the channel
y_pred = torch.argmax(y_pred, dim=1)  # Get classes with the highest probability

# Convert tensors to numpy arrays
y_real = Y.cpu().numpy()
y_pred = y_pred.cpu().numpy()</code></pre>

<h2>Confusion Matrix</h2>
<h2>Matriz de Confusión</h2>

<p>The confusion matrix is calculated and displayed.</p>
<p>Se calcula y muestra la matriz de confusión.</p>

<pre><code># Calculate confusion matrix
cm = confusion_matrix(y_real, y_pred, labels=np.arange(10))

# Configure figure and heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
             xticklabels=np.arange(10), yticklabels=np.arange(10))

# Set titles and labels
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Prediction', fontsize=14)
plt.ylabel('Real', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()</code></pre>

<h2>Model Saving</h2>
<h2>Guardar Modelo</h2>

<p>The model parameters and configuration are saved to files.</p>
<p>Se guardan los parámetros del modelo y la configuración en archivos.</p>

<pre><code>torch.save(model.state_dict(), 'model_state.pth')

model_config = {
    'layers': [
        {'type': 'Conv2d', 'params': {'in_channels': 1, 'out_channels': 6, 'kernel_size': 5, 'padding': 1}},
        {'type': 'ReLU', 'params': {}},
        {'type': 'MaxPool2d', 'params': {'kernel_size': 2, 'stride': 2}},
        {'type': 'Conv2d', 'params': {'in_channels': 6, 'out_channels': 16, 'kernel_size': 5, 'padding': 1}},
        {'type': 'ReLU', 'params': {}},
        {'type': 'MaxPool2d', 'params': {'kernel_size': 2, 'stride': 2}},
        {'type': 'Conv2d', 'params': {'in_channels': 16, 'out_channels': 120, 'kernel_size': 5}},
        {'type': 'ReLU', 'params': {}},
        {'type': 'Flatten', 'params': {}},
        {'type': 'Linear', 'params': {'in_features': 120, 'out_features': 84}},
        {'type': 'ReLU', 'params': {}},
        {'type': 'Linear', 'params': {'in_features': 84, 'out_features': 10}},
        {'type': 'LogSoftmax', 'params': {'dim': -1}}
    ]
}

with open('model_config.json', 'w') as json_file:
    json.dump(model_config, json_file, indent=4)</code></pre>


