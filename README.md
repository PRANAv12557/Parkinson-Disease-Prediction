# Parkinson's Disease Prediction App 🧠 🔍

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)<br><br>

Parkinson's disease (PD) is a neurodegenerative disorder that affects movement control. This project leverages **machine learning** techniques to predict the likelihood of an individual having Parkinson's disease based on their medical features. 

We have heavily improved and expanded upon a base machine learning model by constructing a **Premium Streamlit web application** that offers **fully automated acoustic feature extraction**, batch processing capabilities, and a sleek, modern user interface. ✨

## 🌟 Key Improvements & New Features

We have transformed this project from a standard ML script into a clinical-grade application by adding the following major advancements:

1. **🎤 Automated Audio Feature Extraction**
   - Users no longer need to manually enter 22 complex acoustic features.
   - Built an integration with `praat-parselmouth` to automatically extract all necessary features (Jitter, Shimmer, HNR, APQ, etc.) directly from a `.wav` voice recording in seconds.
   - Completely automates the prediction pipeline.

2. **📊 Batch CSV Processing**
   - Added a dedicated pipeline for hospitals and researchers to upload a CSV dataset containing multiple patient records.
   - The system instantly runs predictions on the entire batch and generates a downloadable CSV report with the results.

3. **📄 Single Patient Report Auto-Fill**
   - Enhanced the manual input tab to allow uploading a single patient's clinical CSV report.
   - Automatically parses the file and auto-fills the 22 manual input boxes, allowing doctors to do a visual review or "What-If" analysis before running the diagnostics.

4. **🎨 Premium UI/UX Redesign**
   - Completely overhauled the frontend using custom CSS and Streamlit components.
   - Introduced dynamic Plotly graphs for Feature Importance Visualization, showing exactly *why* the model made its prediction.
   - Designed a clean, professional dashboard with statistical cards, a dedicated educational Home Page, and an interactive FAQ section.

---

## Project Structure 🛠️

The project is organized into different components, including model training, audio processing, and the Streamlit web app:

```text
├── model_files/           # Folder containing model files
│   ├── parkinson_model.pkl  # Saved SVM model in pickle format
│   ├── pca.pkl              # Principal Component Analysis (PCA) model
│   ├── scaler.pkl           # StandardScaler model
|
├── data/                  # Folder for dataset and related files
│   └── parkinsons.data    # Original Dataset
│   └── CSV Files/         # Sample single-patient reports for testing
|
├── app8.py                # Main enhanced Streamlit application
├── audio_processing.py    # Custom script for Praat automated feature extraction
├── requirements.txt       # Python dependencies required to run the project
```

## Machine Learning Models Trained & Evaluated 🧑💻

The following Machine Learning models were evaluated on the dataset: <br>
1️⃣ Logistic Regression <br>
2️⃣ Random Forest Classifier <br>
3️⃣ Decision Tree Classifier <br>
4️⃣ Support Vector Machine Classifier (SVM - **Selected**) <br>
5️⃣ Naive Bayes Classifier <br>
6️⃣ K Nearest Neighbor Classifier <br>

*(Model evaluation metrics led to the selection of SVM coupled with PCA and Standard Scaling for maximum clinical accuracy).*

## Installation & Setup 🛠️

### 1. Clone the repository

```bash
git clone https://github.com/your-username/parkinson-disease-prediction.git
cd parkinson-disease-prediction
```

### 2. Set up a virtual environment (Recommended)

#### Using `venv` (Windows):

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### Using `venv` (Mac/Linux):
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
*(Ensure `praat-parselmouth` and `plotly` install correctly for the advanced features to function).*

## Using the Streamlit App 🚀

To interact with the enhanced web app:

1. Navigate to the project directory.
2. Run the application:

   ```bash
   streamlit run app8.py
   ```

3. Your browser will automatically open to `http://localhost:8501`.

### Core Application Capabilities
- **Audio Tab:** Upload a `.wav` file of a sustained "Ahhhh" vowel sound for instant, frictionless prediction.
- **Batch Tab:** Upload a CSV of multiple patients to receive a downloadable prediction report.
- **Manual Tab:** Upload a single-patient CSV report to auto-fill the form, tweak variables, and run diagnostics.

## Acknowledgements 🙏

- Dataset Provider: [UCI Machine Learning Repository - Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Foundational audio processing logic utilizing [Parselmouth](https://parselmouth.readthedocs.io/).
- Built using `streamlit`, `scikit-learn`, `pandas`, and `plotly`.

## Group Details 👩💻👨💻

This project was developed and majorly enhanced by a group of 4 students from **VIT Pune**, under the **CSAI-B** branch.

| **Roll Number** | **Official Name**              |
|-----------------|--------------------------------|
| 33              | Shrey Santosh Rupnavar         |
| 37              | Salitri Atharva Akhil          |
| 60              | Tanishq Sudhir Thuse           |
| 61              | Tripti Prakash Mirani          |

---
#### If you have any questions or suggestions, feel free to open an issue or reach out directly! 😄👋ishq Sudhir Thuse           |
| 61              | Tripti Prakash Mirani          |


#### If you have any questions or suggestions, feel free to open an issue or reach out directly! 😄👋



