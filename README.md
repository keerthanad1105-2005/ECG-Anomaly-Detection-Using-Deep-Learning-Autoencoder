# ECG Anomaly Detection Using Deep Learning Autoencoder

## 📌 Project Overview

Electrocardiogram (ECG) signals provide important information about the electrical activity of the heart and can be used to identify abnormal cardiac patterns. Manual analysis of ECG signals can be time-consuming and requires medical expertise.

This project implements a **Deep Learning-based ECG Anomaly Detection system using an Autoencoder**. The model is trained primarily on normal ECG patterns and learns to reconstruct them. When an ECG signal differs significantly from the learned normal pattern, the reconstruction error increases, allowing the signal to be identified as anomalous.

The project uses **TensorFlow/Keras** to build and train the Autoencoder and **Scikit-learn** to evaluate its detection performance.

---

## 🎯 Objectives

* To develop a deep learning model for ECG anomaly detection.
* To preprocess and normalize ECG data before model training.
* To learn normal ECG patterns using an Autoencoder.
* To identify anomalous ECG signals using reconstruction error.
* To calculate an anomaly detection threshold from training reconstruction errors.
* To evaluate the model using accuracy, precision, and recall.
* To visualize normal and anomalous ECG signals and their reconstructed outputs.

---

## 🧠 Methodology

The overall workflow of the project is:

```text
ECG Dataset
     ↓
Data Loading
     ↓
Train-Test Split
     ↓
Data Normalization
     ↓
Separate Normal and Anomalous ECG Data
     ↓
Train Autoencoder on Normal ECG Data
     ↓
Reconstruct ECG Signals
     ↓
Calculate Reconstruction Error
     ↓
Determine Anomaly Threshold
     ↓
Normal / Anomalous Prediction
     ↓
Model Evaluation
```

---

## 📊 Dataset

The project uses an ECG dataset provided through the TensorFlow ECG dataset source.

The dataset is loaded directly using:

```python
pd.read_csv(
    "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv",
    header=None
)
```

The final column of the dataset contains the labels, while the remaining columns represent ECG signal measurements.

Each ECG sample contains **140 data points**.

The dataset is divided into training and testing sets using an **80:20 split** with a fixed random state of 21.

> The dataset itself is not included in this repository. The notebook downloads it during execution.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

### 1. Dataset Loading

The ECG dataset is loaded into a Pandas DataFrame and converted into a NumPy array.

### 2. Separation of Data and Labels

The final column is extracted as the label, while the remaining 140 values represent the ECG data.

### 3. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

### 4. Normalization

The ECG values are normalized using the minimum and maximum values calculated from the training data:

```text
Normalized Value =
(Value - Minimum) / (Maximum - Minimum)
```

The normalized data is then converted to TensorFlow `float32` tensors.

### 5. Normal and Anomalous Samples

The labels are converted into Boolean values to separate:

* Normal ECG samples
* Anomalous ECG samples

The Autoencoder is trained using the normal training ECG samples.

---

## 🤖 Deep Learning Model

This project uses a **Deep Learning Autoencoder**, rather than a CNN-based classification architecture.

An Autoencoder consists of two major components:

```text
Input ECG
    ↓
Encoder
    ↓
Compressed Representation
    ↓
Decoder
    ↓
Reconstructed ECG
```

### Encoder

The encoder compresses the input ECG signal into a lower-dimensional representation.

Architecture:

```text
Input: 140 features
      ↓
Dense Layer: 32 neurons
      ↓
Dense Layer: 16 neurons
      ↓
Dense Layer: 8 neurons
```

ReLU activation is used in the hidden layers.

### Decoder

The decoder reconstructs the original ECG signal from the compressed representation.

Architecture:

```text
8 neurons
   ↓
Dense Layer: 16 neurons
   ↓
Dense Layer: 32 neurons
   ↓
Dense Layer: 140 neurons
   ↓
Reconstructed ECG
```

The output layer uses a sigmoid activation function.

---

## 🔧 Model Configuration

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Model             | Deep Learning Autoencoder |
| Framework         | TensorFlow / Keras        |
| Input Size        | 140                       |
| Encoder Layers    | 32 → 16 → 8               |
| Decoder Layers    | 16 → 32 → 140             |
| Hidden Activation | ReLU                      |
| Output Activation | Sigmoid                   |
| Optimizer         | Adam                      |
| Loss Function     | Mean Absolute Error (MAE) |
| Epochs            | 20                        |
| Batch Size        | 512                       |

---

## 🏋️ Model Training

The Autoencoder is trained using **normal ECG training samples**.

The model attempts to reconstruct the input ECG signal:

```text
Normal ECG → Encoder → Decoder → Reconstructed ECG
```

The difference between the original ECG and reconstructed ECG is measured using **Mean Absolute Error (MAE)**.

The model is trained for **20 epochs** with a batch size of **512**.

---

## 🚨 Anomaly Detection

The main principle of the project is **reconstruction error**.

A well-trained Autoencoder should reconstruct ECG patterns similar to the normal ECG signals on which it was trained.

For an anomalous ECG signal, the reconstruction may be less accurate, resulting in a larger reconstruction error.

The project calculates the reconstruction loss using Mean Absolute Error.

### Threshold Calculation

The anomaly threshold is calculated as:

```text
Threshold = Mean Training Loss + Standard Deviation of Training Loss
```

The threshold is then used to classify ECG samples.

Conceptually:

```text
Reconstruction Error
        ↓
   Compare with
     Threshold
        ↓
 ┌───────────────┐
 │               │
Below          Above
Threshold      Threshold
 │               │
Normal        Anomalous
```

---

## 📈 Model Evaluation

The project evaluates the anomaly detection model using:

* Accuracy
* Precision
* Recall

The evaluation metrics are calculated using **Scikit-learn**.

The notebook also generates visualizations for:

* Normal ECG signals
* Anomalous ECG signals
* Training loss
* Validation loss
* Reconstructed normal ECG
* Reconstructed anomalous ECG
* Reconstruction error distributions

---

## 📊 Visualization

The project visualizes the original and reconstructed ECG signals to understand how effectively the Autoencoder learns the ECG patterns.

### Normal ECG

The original normal ECG signal is compared with its reconstructed version.

### Anomalous ECG

The anomalous ECG signal is also reconstructed and compared with the original signal.

The difference between the original and reconstructed signals provides a visual representation of reconstruction error.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Deep Learning Autoencoder

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Data Visualization

* Matplotlib

### Development Environment

* Jupyter Notebook

---

## 📁 Project Structure

```text
ECG-Anomaly-Detection/
│
├── ECG_Anomaly_Detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### File Description

| File                          | Description                                                                                                 |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ECG_Anomaly_Detection.ipynb` | Main notebook containing data preprocessing, model development, training, anomaly detection, and evaluation |
| `README.md`                   | Project documentation                                                                                       |
| `requirements.txt`            | Python dependencies required to run the project                                                             |
| `.gitignore`                  | Files and folders excluded from Git tracking                                                                |

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ECG-Anomaly-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd ECG-Anomaly-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Open the project folder.
2. Start Jupyter Notebook:

```bash
jupyter notebook
```

3. Open:

```text
ECG_Anomaly_Detection.ipynb
```

4. Run the notebook cells sequentially.

The notebook will:

1. Load the ECG dataset.
2. Split the data into training and testing sets.
3. Normalize the ECG values.
4. Separate normal and anomalous samples.
5. Build the Autoencoder.
6. Train the model using normal ECG data.
7. Calculate reconstruction errors.
8. Determine the anomaly threshold.
9. Predict normal and anomalous ECG samples.
10. Calculate evaluation metrics.
11. Generate ECG and training visualizations.

---

## 📋 Requirements

Create a `requirements.txt` file containing:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

---

## 🔍 Key Features

* ECG signal preprocessing
* Min-max normalization
* Deep Learning-based anomaly detection
* Autoencoder architecture
* Normal ECG pattern learning
* Reconstruction-based anomaly detection
* Automatic threshold calculation
* ECG reconstruction visualization
* Training and validation loss visualization
* Accuracy, precision, and recall evaluation

---

## 🌍 Applications

The approach demonstrated in this project can be useful as a foundation for:

* Automated ECG screening
* Cardiac signal monitoring
* Abnormal ECG pattern detection
* Healthcare decision-support systems
* Remote patient monitoring
* Research in biomedical signal processing

> This project is intended for educational and research purposes and should not be considered a standalone medical diagnostic system.

---

## 🚀 Future Enhancements

The project can be further improved by:

* Using larger and more diverse ECG datasets.
* Experimenting with CNN-based ECG models.
* Exploring LSTM and hybrid CNN-LSTM architectures.
* Performing multi-class arrhythmia classification.
* Improving anomaly threshold selection.
* Adding additional evaluation metrics such as F1-score and ROC-AUC.
* Saving the trained model for independent prediction.
* Developing a dedicated prediction interface.
* Evaluating the model on clinically validated datasets.

---

## 📌 Limitations

* The current implementation focuses on **normal versus anomalous ECG detection** rather than identifying specific arrhythmia types.
* The model is trained using an Autoencoder architecture rather than a CNN classifier.
* The current notebook does not provide a web application.
* The project does not represent a clinically validated diagnostic system.

---

## 👩‍💻 Author

**Keerthana D**

BE – Information Science and Engineering

---

## ⭐ Project Highlights

**Domain:** Deep Learning / Healthcare / Biomedical Signal Processing

**Model:** Deep Learning Autoencoder

**Task:** ECG Anomaly Detection

**Framework:** TensorFlow / Keras

**Language:** Python

**Environment:** Jupyter Notebook

---

## 📄 License

This project is intended for educational and research purposes. You may modify and extend the project for learning and experimentation with appropriate attribution.
