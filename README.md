# **NN Classification**

## Multi Labels Classification

Implementation of a model of feedforward neural network designed to classify a synthetic 2D dataset with four distinct classes based on two features

---
<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/b9cf96ba-3e80-48ed-a4cb-5a85d5dc8b97" width="75%" />
  <br />
  <strong>Figure 1: data used</strong>
</p>
<p float="left" style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/b9ce07de-bc5c-49f1-961e-cba5004abb9c" width="75%" />
  <br />
  <strong>Figure 2: model prediction after training</strong>
</p>

## **Model Details**

- **Architecture**: Feedforward neural network with two hidden layers (`nn.Linear`), 2 input features, 4 output classes.
- **Loss Function**: **Cross-Entropy Loss** (for classification tasks).
- **Optimizer**: **SGD** (Stochastic Gradient Descent) learning rate of 0.1.
- **Training**: 100 epochs with forward propagation, backpropagation, and weight updates.

---

## **Training and Evaluation**
The training script:
1. Splits data into train (80%) and test (20%) sets.
2. Trains the model using the train set.
3. Evaluates performance on the test set every 100 epochs.

---

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/feedforward-network-pytorch.git
   cd feedforward-network-pytorch

2. **Load the model in your code**:

```python
from model import BlobModel
import torch

# Initialize the model (same architecture as during training)
model = BlobModel(input_features=2, output_features=4, hidden_units=8)

# Load the pre-trained model weights
model.load_state_dict(torch.load("models/multi_labels_classification.pth"))

# Set the model to evaluation mode
model.eval()
```

><details>
>  <summary><i>Binary Classification with Circles Dataset</i></summary>
>
>Implementation of a feedforward neural network model designed to classify a synthetic 2D dataset with two features into two classes (binary classification).
>
>---
><p float="left" style="text-align: center; margin-right: 10px;">
>  <img src="https://github.com/user-attachments/assets/07749555-6dc7-466d-83b4-134a7d993625" width="75%" />
>  <br />
>  <strong>Figure 1: Data used</strong>
></p>
><p float="left" style="text-align: center;">
>  <img src="https://github.com/user-attachments/assets/a81b49ea-fd77-4c40-9b4b-9b0243928b05" width="75%" />
>  <br />
>  <strong>Figure 2: Model prediction after training</strong>
></p>
>
>## **Model Details**
>
>- **Architecture**: Feedforward neural network with two hidden layers (`nn.Linear`), 2 input features, 1 output class (binary classification).
>- **Activation Function**: **ReLU** (Rectified Linear Unit).
>- **Loss Function**: **Binary Cross-Entropy Loss** (`BCEWithLogitsLoss`).
>- **Optimizer**: **SGD** (Stochastic Gradient Descent) with a learning rate of 0.1.
>- **Training**: 1000 epochs with forward propagation, backpropagation, and weight updates.
>
>---
>
>## **Training and Evaluation**
>The training script:
>1. Splits data into train (80%) and test (20%) sets.
>2. Trains the model using the train set.
>3. Evaluates performance on the test set every 10 epochs.
>
>---
>
>### **Setup**
>1. Clone the repository:
>   ```bash
>   git clone https://github.com/yourusername/feedforward-network-pytorch.git
>   cd feedforward-network-pytorch
>   ```
>2. **Load the model in your code**:
>   ```python
>   from model import CircleModelV0
>   import torch
>   
>   # Initialize the model (same architecture as during training)
>   model = CircleModelV0()
>   
>   # Load the pre-trained model weights
>   model.load_state_dict(torch.load("models/binary_classification.pth"))
>   
>   # Set the model to evaluation mode
>   model.eval()
>  ```
></details>
