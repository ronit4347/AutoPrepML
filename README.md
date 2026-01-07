# 👨🏻‍💻 AutoPrepML

**From Raw Data to ML‑Ready — Automatically**

AutoPrepML is an end‑to‑end **automatic data preprocessing system** that converts raw datasets into clean, machine‑learning‑ready data in seconds. It combines **Auto EDA**, **smart preprocessing**, and a **clean Streamlit UI** so users can focus on modeling, not cleaning.

---

## 🧠 Why AutoPrepML?

In real‑world data science:

* ⏳ **70–80% of time** is spent on preprocessing
* Beginners struggle with missing values, encoding, scaling, outliers, and splits
* Many projects jump straight to modeling ❌

**AutoPrepML solves this gap.**

---

## ✨ Key Features

### 📤 Dataset Upload

* Supports **CSV** and **Excel** files
* Automatic detection of:

  * Numeric vs categorical columns
  * Dataset size

---

### 📊 Auto EDA (Before Preprocessing)

* Missing value analysis
* Target‑aware checks
* Correlation with target (numeric)
* Compact, readable visualizations

---

### ⚙️ Smart Automatic Preprocessing

* ✔ Removes duplicate rows
* ✔ Handles missing values:

  * Numeric → **Median**
  * Categorical → **Most frequent**
* ✔ Normalizes column names
* ✔ Automatically detects column types

---

### 🔥 Intelligent Outlier Handling

* Uses **IQR method**
* Caps extreme values (no row deletion)
* Shows:

  * Outliers **before** preprocessing
  * Outliers **after** preprocessing

---

### 🔄 Feature Transformation

* **Categorical Encoding**: One‑Hot Encoding
* **Scaling**: StandardScaler for numeric features
* **Pipeline‑based** (leakage‑safe)

---

### 🧪 Supervised & Unsupervised Modes

* ✅ Valid target selected → **Supervised ML**
* ⚠ Invalid / missing / ID‑like target → **Unsupervised mode**
* Protects against **data leakage** automatically

#### 🥈 Before vs After Comparison

* Outliers reduced
* Feature count increased
* Clear raw vs processed comparison

---

### 📦 ML‑Ready Output

Depending on mode:

**Supervised Mode**

* `X_train.csv`
* `X_test.csv`
* `y_train.csv`
* `y_test.csv`

**Unsupervised Mode**

* `X_train.csv`

All outputs are:

* Encoded
* Scaled
* Ready for modeling

---

## 🎨 User Interface

* Built with **Streamlit**
* Clean tab‑based layout:

  * Raw Data
  * Auto EDA
  * Preprocessing
  * ML‑Ready Data
  * Downloads
* Progress indicators & clear feedback

---

## 🛠 Tech Stack

* Python
* Streamlit
* Pandas, NumPy
* Scikit‑learn
* Matplotlib, Seaborn
* Joblib

---

## 📂 Project Structure

```
AutoPrepML/
│
├── app.py                # Streamlit UI
├── preprocessing.py      # Core preprocessing engine
├── requirements.txt
├── README.md
└── autoprepml_pipeline.pkl
```

---

## HOW TO USE
link = https://autoprepml-4347.streamlit.app/


## 🎯 Use Cases

* Data science beginners
* Hackathons & competitions
* Rapid ML prototyping
* Teaching preprocessing concepts
* AutoML preprocessing layer

---

## 🏆 Why This Project Stands Out

* Solves a **real data science pain point**
* Strong preprocessing logic
* Clean UX (often ignored in competitions)
* Prevents data leakage
* Handles messy real‑world datasets

---

## 🔮 Future Enhancements

* Auto model training
* Feature importance
* Auto target suggestion
* AutoML integration
* Cloud deployment

---

## 🤝 Author

Built with ❤️ by ** RONIT PATANKAR **

> *“Good models fail on bad data. AutoPrepML fixes the data first.”*
