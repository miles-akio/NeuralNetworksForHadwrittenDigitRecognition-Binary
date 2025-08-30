# NeuralNetworksForHadwrittenDigitRecognition-Binary


````markdown
# Lab: Neural Networks for Handwritten Digit Recognition (Binary)

This lab demonstrates how to build and train a neural network to recognize handwritten digits **0 and 1** using both **TensorFlow/Keras** and **NumPy** implementations. The exercise covers dataset exploration, model representation, TensorFlow implementation, and forward propagation in NumPy.

---

## 📌 Outline
1. **Packages**
2. **Neural Networks**
   - 2.1 Problem Statement  
   - 2.2 Dataset  
   - 2.3 Model Representation  
   - 2.4 TensorFlow Model Implementation *(Exercise 1)*  
   - 2.5 NumPy Model Implementation (Forward Propagation) *(Exercise 2)*  
   - 2.6 Vectorized NumPy Model Implementation *(Optional, Exercise 3)*  
   - 2.7 Congratulations!  
   - 2.8 NumPy Broadcasting Tutorial *(Optional)*  

---

## ⚙️ 1 - Packages
The following Python libraries are required:
- **NumPy** – scientific computing and matrix operations  
- **Matplotlib** – visualization of images and results  
- **TensorFlow / Keras** – neural network implementation  

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *


---

## 🧠 2 - Neural Networks

### 2.1 Problem Statement

* Task: **Binary classification** to recognize handwritten digits (0 or 1).
* Applications: Zip code recognition, bank check processing, etc.
* Extension: This will later expand to classify all **10 digits (0–9)**.

---

### 2.2 Dataset

* Subset of the **MNIST dataset**.
* **1000 training examples**, each a **20×20 grayscale image** (flattened into 400 features).
* Labels:

  * `y = 0` → digit is **0**
  * `y = 1` → digit is **1**

**Shapes:**

* `X.shape = (1000, 400)`
* `y.shape = (1000, 1)`

Visualization is done by reshaping rows of `X` into 20×20 grids.

---

### 2.3 Model Representation

* Input layer: **400 features (20×20 pixels)**
* Hidden Layer 1: **25 units**, sigmoid activation
* Hidden Layer 2: **15 units**, sigmoid activation
* Output Layer: **1 unit**, sigmoid activation

**Parameter Dimensions:**

* `W1: (400, 25)`, `b1: (25,)`
* `W2: (25, 15)`, `b2: (15,)`
* `W3: (15, 1)`, `b3: (1,)`

---

### 2.4 TensorFlow Model Implementation *(Exercise 1)*

Implemented with **Keras Sequential API**:

```python
model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(25, activation='sigmoid'),
    Dense(15, activation='sigmoid'),
    Dense(1, activation='sigmoid')
], name="my_model")
```

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam (learning rate = 0.001)
* **Training**: 20 epochs

**Example Predictions:**

* Zero → `prediction ≈ 0.01`
* One → `prediction ≈ 0.97`

---

### 2.5 NumPy Model Implementation *(Exercise 2)*

Forward propagation implemented manually with NumPy:

```python
def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(a_in, W[:, j]) + b[j]
        a_out[j] = g(z)
    return a_out
```

* Inputs: activation `a_in`, weight matrix `W`, bias `b`, activation function `g`.
* Output: activation values `a_out`.

---

### 2.6 Vectorized NumPy Implementation *(Optional, Exercise 3)*

* Uses `np.matmul()` for matrix multiplication.
* Eliminates the explicit loop over units for faster computation.

---

### 2.7 Congratulations!

At this stage, you’ve:

* Explored the dataset.
* Implemented and trained a **Keras neural network**.
* Built your own **NumPy-based forward propagation layer**.
* Learned both **loop-based** and **vectorized** implementations.

---


## 🔗 References

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* TensorFlow & Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

```

````
