# -Lab-2.1-Taming-Gradients-Capacity-
This README covers the challenges of training **Deep Neural Networks**, specifically focusing on how gradient behavior and model capacity affect learning.

---

As networks get deeper, they become harder to train due to numerical instabilities and the risk of memorizing data rather than learning it. This lab explores **Vanishing Gradients**, **Initialization Strategies**, and the balance between **Underfitting** and **Overfitting**.

## 🧬 Core Concepts

### 1. The Vanishing Gradient Problem
When using certain activation functions like **Sigmoid** in deep networks, gradients tend to get smaller as they propagate back from the output layer to the input layer. By the time they reach the first layers, they are nearly zero, effectively stopping those layers from learning.



### 2. Smart Initialization (He vs. Xavier)
To combat gradient issues, we use specialized weight initialization:
* **Xavier (Glorot) Initialization:** Best for symmetric activations like `tanh` or `sigmoid`.
* **He (Kaiming) Initialization:** Designed specifically for `ReLU` to account for the fact that half of the neurons are "off" at any given time.

### 3. Capacity & Generalization
We experiment with how the size of the network (depth and width) impacts its ability to learn:
* **Small Capacity:** A shallow, narrow model may **underfit**, failing to capture the complexity of the FashionMNIST dataset.
* **High Capacity + Small Data:** A deep, wide model trained on a tiny subset (1,000 images) will **overfit**, achieving near-perfect training accuracy but performing poorly on the validation set.

---

## 🔬 Experiments Conducted

### Experiment A: Activation & Gradient Profiles
We track the **norm of the gradients** across different layers during training.
* **Sigmoid Profile:** Usually shows a sharp decline in gradient strength in earlier layers (Vanishing Gradient).
* **ReLU + He Profile:** Maintains a more consistent gradient flow, allowing deeper networks to converge.

### Experiment B: Overfitting & Regularization
We simulate a classic overfitting scenario and apply a fix:
* **The Problem:** Training a "Big" model on "Tiny" data leads to a massive gap between training loss and validation loss.
* **The Fix (L2 Regularization):** By adding `weight_decay` to the Adam optimizer, we penalize large weights. This forces the model to be simpler and improves its ability to generalize to new data.

---

## 🛠️ Implementation Details

* **Model:** `DeepMLP` — A flexible class where you can toggle `depth`, `width`, `activation`, and `initialization`.
* **Logging:** The `train_one_epoch` function captures the Euclidean norm of weights at every layer to visualize the "Gradient Profile."
* **Data Splitting:** Uses a dedicated **Validation Set** ($5,000$ images) to monitor generalization during training, separate from the final Test Set.

---

## 📊 Summary of Results
| Configuration | Observation |
| :--- | :--- |
| **Sigmoid (8 Layers)** | High training loss; gradients disappear in early layers. |
| **ReLU + He (8 Layers)** | Fast convergence; stable gradients throughout. |
| **Big Model / Tiny Data** | Training accuracy $\approx 100\%$, Validation accuracy is low. |
| **Big Model + L2 Decay** | Training accuracy drops slightly, but Validation accuracy increases (Better Generalization). |

---

## 🚀 How to Run
1.  Ensure `torch`, `torchvision`, and `matplotlib` are installed.
2.  Run the script to see the gradient profile plots.
3.  Observe the console output to compare how **L2 Regularization** tames the "Big" model's tendency to overfit.
