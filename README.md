# Brain Tumor Detection Model

This project presents a Convolutional Neural Network (CNN)–based approach to classify MRI scans as tumor-positive or normal. The model is implemented using PyTorch and trained on a labeled MRI dataset. It was developed by **Shahad Hossain**, a student at the *College of Staten Island*, under the mentorship of **Distinguished Professor Sos Agaian**.

## 🧠 Project Overview

Brain tumor detection from MRI scans is a critical task in medical diagnostics. This model utilizes deep learning to automate tumor classification with high accuracy using a binary classification framework (tumor vs. normal).

## 📁 Repository Structure

```
Brain-Tumor-Detection-Model/
├── brain_tumor_model.pth       # Trained PyTorch model
├── dataset.csv                 # Helper CSV to dynamically load dataset
├── requirements.txt            # Python dependencies
├── test_image.jpeg             # Sample MRI scan for model testing
└── model.ipynb                 # Jupyter notebook with full training & testing code
```

## 🧪 Dataset

The model is trained on the **PMRAM Bangladeshi Brain Cancer - MRI Dataset**, which includes the following classes:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **Normal (non-tumor)**

For classification, the dataset is grouped into:
- `Tumor`: Glioma, Meningioma, Pituitary
- `Normal`: Healthy brain scans

Images are preprocessed to 256×256 resolution and augmented for performance.

## 🏗️ Model Architecture

A custom CNN was implemented with:
- 3 convolutional layers with ReLU and max pooling
- Fully connected layers with sigmoid activation for binary classification
- Final output: probability of tumor presence (threshold > 0.5)

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Shahad-Hossain/Brain-Tumor-Detection-Model.git
cd Brain-Tumor-Detection-Model

# Create environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the model

Open `model.ipynb` and run all cells to:
- Load and preprocess the data
- Train the CNN for 10 epochs
- Save the model to `brain_tumor_model.pth`

### 2. Inference on new MRI scan

The notebook performs inference using `test_image.jpeg`. You can test out the model by using your own MRI images and viewing the model's accuracy.

```python
output = model(input_tensor)
prediction = (output.item() > 0.5)
```

**Output:**
```
Prediction: Tumor detected
```
or
```
Prediction: No tumor detected (Normal)
```

## 📊 Evaluation

The model reports training accuracy at each epoch. Accuracy on the test split is computed using standard binary metrics. Final accuracy exceeds **90%** on the validation data.

## 🔬 Technologies Used

- Python 3
- PyTorch
- NumPy, Matplotlib, Scikit-learn
- PIL, torchvision

## 👨‍🎓 Authors & Acknowledgments

- **Author:** Shahad Hossain (College of Staten Island)
- **Supervisor:** Prof. Sos Agaian (Distinguished Professor, CUNY)
- **Dataset:** PMRAM Bangladeshi Brain Cancer MRI Dataset

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> For educational and research purposes only. Always consult with medical professionals before making health-related decisions.