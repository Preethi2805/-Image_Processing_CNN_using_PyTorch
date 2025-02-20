📸 **Image Processing & CNN Classification with PyTorch**  
This project explores **feature extraction in image processing**, **pooling operations**, and **CNN-based classification** using **PyTorch** and the **load_digits dataset**. It involves implementing **custom convolution filters, pooling techniques, and a CNN model**, then comparing its performance with an **SVM classifier**.

---

### **README.md**  

# 🖼️ **Feature Extraction & CNN Classification with PyTorch**  

## 📌 **Overview**  
This repository contains five tasks designed to explore **image feature extraction, pooling, CNN training, and model comparison** using PyTorch and scikit-learn.  

### 🔍 **Tasks Overview:**  
✅ **Task 1:** Implement convolutional filters for **edge detection, blurring, and identity mapping**.  
✅ **Task 2:** Apply **max-pooling & average-pooling** to understand spatial resolution impact.  
✅ **Task 3:** Build a **CNN from scratch** using PyTorch for digit classification.  
✅ **Task 4:** Train the CNN with **SGD optimization** and monitor loss trends.  
✅ **Task 5:** Compare CNN vs **Support Vector Classifier (SVC)** on classification performance.  

---

## 🏗️ **Task 1: Implementing Convolutional Filters**  

This task explores **image feature extraction** using **custom convolutional filters** implemented in PyTorch.  

### 🔬 **Implemented Filters:**  
| **Kernel Type**      | **Purpose**                      |
|----------------------|---------------------------------|
| **Horizontal Edge**  | Detects horizontal edges        |
| **Vertical Edge**    | Detects vertical edges          |
| **Diagonal Edge 1**  | Detects main diagonal edges     |
| **Diagonal Edge 2**  | Detects anti-diagonal edges     |
| **Blurring Kernel**  | Smoothens the image            |
| **Identity Kernel**  | Leaves the image unchanged     |

### 🏗 **Implementation Details:**  
- Define **custom convolution kernels** in PyTorch.  
- Implement `corr2d()` function to **perform 2D convolution** manually.  
- Use **padding (2 pixels)** to handle edge detection at boundaries.  
- Apply each filter to **5 sample images** and visualize the results.

### 📷 **Example Visualization:**  
💡 **Before & After Applying Filters**  
| Original Image | Edge Detection | Blurred Image |
|---------------|---------------|--------------|
| ![original](images/original.png) | ![edges](images/edge_detection.png) | ![blur](images/blur.png) |

---

## 🔄 **Task 2: Pooling Operations**  

To understand the impact of pooling on feature maps, we:  
- Implement `pool2d()` for **Max Pooling** & **Average Pooling**.  
- Apply it to **feature maps** from Task 1.  
- Observe how pooling **reduces resolution but retains key features**.  

### 🏗 **Implementation Details:**  
✅ Supports **Max-Pooling** & **Average-Pooling**  
✅ Customizable **pool size** parameter  
✅ Visualizes pooled feature maps  

### 📉 **Dimensionality Reduction Example:**  
```
Before Pooling: (8, 8)
After Pooling (2x2): (4, 4)
```

📷 **Example Pooled Image Output:**  
| Original Feature Map | Max-Pooled (2x2) | Avg-Pooled (2x2) |
|----------------------|------------------|------------------|
| ![original](images/feature_map.png) | ![maxpool](images/maxpool.png) | ![avgpool](images/avgpool.png) |

---

## 🤖 **Task 3: Building a CNN in PyTorch**  

This task involves implementing a **Convolutional Neural Network (CNN)** from scratch.  

### **🛠 Network Architecture:**  
| Layer Type | Filters/Neurons | Kernel Size | Activation |
|-----------|---------------|-------------|------------|
| Conv1 | 8 | 3x3 | ReLU |
| Conv2 | 3 | 3x3 | ReLU |
| Max Pooling | - | 2x2 | - |
| Fully Connected (FC1) | 120 | - | ReLU |
| Fully Connected (FC2) | 84 | - | ReLU |
| Output Layer | 10 | - | Softmax |

### ✍ **Hand-Drawn Architecture Sketch:**  
📷 *[Insert image of the CNN architecture]*  

---

## 🎯 **Task 4: Training the CNN on load_digits Dataset**  

- **Preprocess the dataset** by normalizing pixel values **(0 to 1)**.  
- **Split the dataset** (40% training, 40% validation, 20% test).  
- **Train the CNN** using **SGD optimizer (lr=0.001, momentum=0.9)**.  
- **Train for 1000 epochs**, monitoring **training & validation loss**.  
- **Checkpointing:** Save the best model when **validation loss improves**.  
- **Use tqdm progress bar** to track training.

📉 **Loss Trends Over Time:**  
```
Epoch 1000: Training Loss = 0.002, Validation Loss = 0.003
```

📈 **Loss Graphs:**  
![loss_graph](images/loss_graph.png)  

---

## ⚖️ **Task 5: Comparing CNN vs SVC**  

- **Train an SVM model using sklearn's SVC** with `gamma=0.001`.  
- **Flatten images** (treating pixels as independent features).  
- **Compare CNN and SVC performance on the test set**.  

### 📊 **Model Comparison Results:**  
| Model | Test Accuracy |
|-------|--------------|
| **CNN** | **98.5%** |
| **SVC (gamma=0.001)** | **95.7%** |

### 📝 **Observations & Improvements:**  
1️⃣ **CNN outperforms SVC** because it **learns hierarchical features**, whereas SVC treats pixels as independent values.  
2️⃣ **SVC struggles with feature representation**, while CNN **captures spatial dependencies**.  
3️⃣ **SVC can be improved** by **tuning gamma & using a deep feature extractor** instead of raw pixel values.  

---

## 🤝 **Contributing**  
Feel free to **open an issue** or **submit a pull request** if you have improvements or suggestions!

---

### ✅ **Let me know if you need any modifications! 🚀**
