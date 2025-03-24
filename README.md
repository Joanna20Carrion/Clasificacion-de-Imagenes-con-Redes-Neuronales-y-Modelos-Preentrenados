# Clasificación de Imágenes con Redes Neuronales y Modelos Preentrenados

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-0.10%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-blueviolet)
![PIL](https://img.shields.io/badge/PIL-Image%20Processing-red)
![LivLossPlot](https://img.shields.io/badge/LivLossPlot-Training%20Monitoring-brightgreen)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNNs-brightgreen)

## Descripción
Este proyecto se centra en la clasificación de imágenes en diferentes categorías utilizando enfoques de aprendizaje profundo. Incluye el uso de:

- **Clasificadores lineales** para tareas iniciales de clasificación.
- **Redes Neuronales Convolucionales (CNNs)** diseñadas desde cero.
- **Modelos preentrenados** como EfficientNet, ajustados para el conjunto de datos utilizado.

El sistema fue entrenado y evaluado con el conjunto de datos SUN20, que incluye 20 categorías de imágenes balanceadas.

---

## 🚀 Estructura del Proyecto

### 1. Clasificación Lineal:
- Implementación de un clasificador lineal simple.
- Entrenamiento utilizando Stochastic Gradient Descent (SGD).
- Visualización de resultados iniciales.

### 2. Redes Neuronales Convolucionales (CNN):
- Definición de arquitecturas convolucionales personalizadas.
- Entrenamiento y evaluación en PyTorch.

### 3. Transfer Learning:
- Ajuste de modelos preentrenados, como EfficientNet.
- Optimizaciones para mejorar la precisión en el conjunto de datos SUN20.

### 4. Exploración y Visualización de Datos:
- Uso de transformaciones en las imágenes.
- Visualización de categorías y resultados en cuadrículas.

---

## 🛠 Tecnologías Utilizadas

- **Lenguaje de programación:** ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
- **Bibliotecas principales:**
  - ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white) **PyTorch** → Para la construcción y entrenamiento de modelos.
  - ![Torchvision](https://img.shields.io/badge/Torchvision-0.10%2B-orange) **Torchvision** → Para el manejo de datos de imagen.
  - ![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-blueviolet) ![PIL](https://img.shields.io/badge/PIL-Image%20Processing-red) **Matplotlib y PIL** → Para la visualización y manipulación de imágenes.
  - ![LivLossPlot](https://img.shields.io/badge/LivLossPlot-Training%20Monitoring-brightgreen) **LivLossPlot** → Para el monitoreo visual de las curvas de pérdida y precisión durante el entrenamiento.
    
---

## ⚙️ Configuración del Entorno

### Requisitos previos
1. Python 3.8 o superior.
2. Instalar las dependencias del proyecto:

```bash
pip install torch torchvision matplotlib livelossplot
```

### Estructura del Dataset
El proyecto utiliza el conjunto de datos **SUN20**, que contiene 20 categorías balanceadas. Para preparar el conjunto de datos:

1. Descarga los archivos del conjunto de datos:

```bash
wget http://www.cs.rice.edu/~vo9/deep-vislang/SUN20-train-sm.tar.gz
wget http://www.cs.rice.edu/~vo9/deep-vislang/SUN20-val.tar.gz
```

2. Extrae los archivos:

```bash
tar -xzf SUN20-train-sm.tar.gz
tar -xzf SUN20-val.tar.gz
```

3. Organiza las carpetas siguiendo la estructura `SUN20/<split>/<categoría>/`.

---

## 🔥 Ejecución del Proyecto

### Entrenamiento de Clasificadores Lineales
1. Configura el dataset y las transformaciones necesarias.
2. Define el modelo lineal en PyTorch:

```python
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
```

3. Entrena el modelo utilizando SGD y analiza las curvas de aprendizaje.

### Redes Neuronales Convolucionales (CNNs)
1. Diseña una arquitectura CNN personalizada:

```python
class ConvnetClassifier(torch.nn.Module):
    def __init__(self):
        super(ConvnetClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, (5, 5))
        self.conv2 = torch.nn.Conv2d(128, 128, (3, 3))
        self.linear1 = torch.nn.Linear(128 * 10 * 10, 256)
        self.linear2 = torch.nn.Linear(256, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), (2, 2)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
``` 

2. Entrena el modelo con funciones de pérdida y optimizadores adecuados.

### Transfer Learning con EfficientNet
1. Carga un modelo preentrenado y ajusta la última capa para adaptarlo al dataset:

```python
from torchvision import models
model = models.efficientnet_v2_m(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 20)
```

2. Entrena el modelo ajustado y evalúa su rendimiento.

---

## 📊 Resultados

- **Clasificadores Lineales:** Precisión inicial aproximada del 25%.
- **Redes Neuronales Convolucionales:** Mejoras en la precisión con arquitecturas personalizadas.
- **Transfer Learning con EfficientNet:** Precisión final de hasta el 91.5% después de 7 épocas.

### Visualización de Resultados

- 📈 Gráficos de pérdida y precisión durante el entrenamiento.
- 📸 Visualización de predicciones con imágenes correctamente clasificadas (en verde) y mal clasificadas (en rojo).

---

## 👤 Autor
**Joanna Alexandra Carrión Pérez**

- 🎓 Bachiller de Ingeniería Electrónica
- 💡 Apasionada por la Ciencia de Datos y la Inteligencia Artificial
- 🔗 ![LinkedIn](https://img.shields.io/badge/LinkedIn-Joanna%20Carrión%20Pérez-blue?style=flat&logo=linkedin) [LinkedIn](https://www.linkedin.com/in/joanna-carrion-perez/)

---

## 📩 Contacto
Para cualquier duda o sugerencia, contáctame a través de **joannacarrion14@gmail.com**.

## 💡 Contribuciones
¡Contribuciones son bienvenidas! Si tienes ideas o mejoras, no dudes en hacer un fork del repositorio y enviar un pull request. 🚀
