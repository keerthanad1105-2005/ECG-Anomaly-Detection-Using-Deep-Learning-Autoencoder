# 🫀 ECG Anomaly Detection Using Deep Learning

### Deep Learning-Based Anomaly Detection for Electrocardiogram (ECG) Signals

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow\&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Network-red?logo=keras\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-013243?logo=numpy\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas\&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Model%20Evaluation-F7931E?logo=scikit-learn\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter\&logoColor=white)

</p>

---

## 🫀 About the Project

This project presents a **Deep Learning-based ECG Anomaly Detection system** using an **Autoencoder neural network**.

The model learns the characteristics of normal ECG signals and reconstructs them. ECG samples with a sufficiently high reconstruction error are identified as anomalous.

The project demonstrates the application of **Deep Learning, ECG signal processing, anomaly detection, and model evaluation** in the healthcare domain.

---

## ✨ Key Features

* 🫀 ECG signal analysis
* 🧠 Deep Learning Autoencoder
* 📊 ECG data normalization
* 🔄 ECG signal reconstruction
* 🚨 Reconstruction-based anomaly detection
* 📈 Training and validation loss visualization
* 📉 Reconstruction error analysis
* 🎯 Automatic anomaly threshold calculation
* 📋 Accuracy, precision, and recall evaluation
* 📓 Jupyter Notebook implementation

---

## 🧠 Model Architecture

```text
                 ECG Input
                    │
                    ▼
             ┌──────────────┐
             │ Dense (32)   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Dense (16)   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Dense (8)    │
             │   Encoder    │
             └──────┬───────┘
                    │
             Compressed ECG
                    │
                    ▼
             ┌──────────────┐
             │ Dense (16)   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Dense (32)   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Dense (140)  │
             │   Decoder    │
             └──────┬───────┘
                    │
                    ▼
             Reconstructed ECG
                    │
                    ▼
          Reconstruction Error
                    │
             ┌──────┴───────┐
             │              │
        Low Error       High Error
             │              │
             ▼              ▼
          Normal         Anomalous
```

---

## 📊 Dataset

The project uses an ECG dataset containing ECG measurements represented by **140 data points per sample**.

The dataset is loaded directly from the TensorFlow dataset source during notebook execution.

The data is divided into:

* **80% Training Data**
* **20% Testing Data**

The Autoencoder is trained using normal ECG samples and subsequently evaluated on both normal and anomalous samples.

---

## ⚙️ Technologies

| Technology          | Purpose                       |
| ------------------- | ----------------------------- |
| 🐍 Python           | Programming language          |
| 🧠 TensorFlow       | Deep Learning framework       |
| 🔥 Keras            | Neural network implementation |
| 🔢 NumPy            | Numerical processing          |
| 🐼 Pandas           | Dataset handling              |
| 📊 Matplotlib       | Data visualization            |
| 🤖 Scikit-learn     | Model evaluation              |
| 📓 Jupyter Notebook | Development environment       |

---

## 📈 Model Configuration

| Parameter         | Configuration |
| ----------------- | ------------- |
| Model             | Autoencoder   |
| Input Size        | 140           |
| Encoder           | 32 → 16 → 8   |
| Decoder           | 16 → 32 → 140 |
| Hidden Activation | ReLU          |
| Output Activation | Sigmoid       |
| Optimizer         | Adam          |
| Loss Function     | MAE           |
| Epochs            | 20            |
| Batch Size        | 512           |

---

## 🔍 How Anomaly Detection Works

The Autoencoder is trained to reconstruct normal ECG signals.

After training, the reconstruction error is calculated using **Mean Absolute Error (MAE)**.

The anomaly threshold is calculated as:

```text
Threshold = Mean Training Loss + Standard Deviation of Training Loss
```

The reconstruction error is then compared with this threshold.

```text
Reconstruction Error
        │
        ▼
   Compare with
     Threshold
        │
   ┌────┴────┐
   │         │
Low Error  High Error
   │         │
   ▼         ▼
 Normal    Anomalous
```

---

## 📊 Evaluation

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**

The project also provides visual analysis through:

* Normal ECG waveform
* Anomalous ECG waveform
* Training loss
* Validation loss
* Reconstructed ECG
* Reconstruction error distribution

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ECG-Anomaly-Detection.git
cd ECG-Anomaly-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

Open:

```text
ECG_Anomaly_Detection.ipynb
```

and execute the cells sequentially.

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

---

## 🔮 Future Enhancements

* Implement CNN-based ECG classification.
* Extend the system to multi-class arrhythmia classification.
* Explore CNN-LSTM and other hybrid architectures.
* Use larger and clinically validated ECG datasets.
* Add additional evaluation metrics such as F1-score and ROC-AUC.
* Develop an independent ECG prediction interface.
* Save and deploy the trained model.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. It is not intended to replace professional medical diagnosis or clinical decision-making.

---

## 👩‍💻 Author

**Keerthana D**

*BE – Information Science and Engineering*

---

### ⭐ If you find this project useful, consider giving it a star!

## 🧪 ECG Anomaly Detection Demo

The project includes a notebook-based demonstration of ECG anomaly detection using a trained Deep Learning Autoencoder.

An ECG sample is passed through the Autoencoder and reconstructed. The reconstruction error is then compared with the calculated anomaly threshold to determine whether the ECG sample is classified as **Normal** or **Anomalous**.

### Demo Output

![ECG Anomaly Detection Demo](Results/ECG_Anomaly_Detection_demo.png)
