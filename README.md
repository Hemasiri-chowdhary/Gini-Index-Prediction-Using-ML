
# Gini Index Prediction & Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🧮 Overview

This project aims to analyze historical data and build a machine learning model to predict the **Gini coefficient** — a metric of income inequality — for different countries/regions over time.  

The Gini coefficient ranges from **0** (perfect equality: everyone’s income is the same) to **1** (perfect inequality: one person has all the income).

---

## 📊 Goals

- Understand how socio-economic factors (GDP, population, income shares, etc.) relate to inequality.  
- Build a predictive model capable of estimating Gini coefficients from input features.  
- Explore model performance, limitations, and potential improvements.

---

## 🛠️ Features

- Data cleaning & preprocessing (handling categorical & numerical features).  
- Encoding categorical variables for use in ML models.  
- Use of decision tree regression to predict Gini with evaluation via metrics like MSE, R².  
- Visualization of trends and relationships in the data (income shares, GDP, etc.).  
- Model serialization for reuse (`gini.pkl`).

---

## 🚀 Technologies & Libraries

- **Python 3.x**  
- Core libraries: `pandas`, `numpy`  
- Visualization: `matplotlib`, `seaborn`  
- Machine learning: `scikit-learn`  
- Notebook environment: Jupyter Notebook  
- Serialization: `pickle` (or joblib, depending on your choice)  

---

## 📥 Data Source

The project uses a dataset (`dataset.xlsx`) which includes:

- Historical values of Gini coefficients  
- Features such as: population, GDP, various income shares (`p1`, `p2`, `p3`, `p4`)  
- Categorical variables: `country`, `area`, `region_wb`, `interpolated`, `subarea`  
- Time span: from mid-20th century onward (varies by country)  

> ⚠️ **Note:** The dataset is **not included** in this GitHub repository (due to size or licensing). You’ll need to get your own copy (e.g. from the World Bank / public data sources), name it appropriately, and ensure it’s placed in the working directory of the notebook.

---

## 📘 Project Structure



Gini–Index-Prediction/
│
├── gini.ipynb             # Jupyter Notebook for analysis, modeling & visuals
├── gini.py                # Optional: Python script version of the notebook workflow
├── LICENSE                # MIT License
├── README.md              # This document
├── gini.pkl               # Serialized trained model (after you run training)
└── dataset.xlsx           # Data file (not included; you must supply it)



---

## 🔧 Installation & Setup

1. Clone the repo:  
   ```bash
   git clone https://github.com/Ashwin-kumar-0309/Gini--Index-Prediction.git
   cd Gini--Index-Prediction


2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
  

3. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, install directly:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Place your dataset file `dataset.xlsx` into the project folder. Adjust file path inside notebook/script if necessary.

5. Run the notebook:

   ```bash
   jupyter notebook gini.ipynb
   ```

---

## 📈 Model Training & Evaluation

* Preprocess data: filter relevant columns, encode categorical variables (`LabelEncoder`), handle missing data.
* Split into training and test sets.
* Train a **Decision Tree Regressor**.
* Evaluate performance using:

  * **Mean Squared Error (MSE)**
  * **R² (Coefficient of Determination)**
* Optionally, serialize the trained model (pickle or similar) so you can load it later without retraining.

---

## 🔍 Results

| Metric | Value         |
| ------ | ------------- |
| MSE    | *e.g.* 0.4803 |
| R²     | *e.g.* 0.9966 |

> These are example values; your results may vary depending on the dataset version, pre-processing, and train/test split.

---

## 🛠️ Potential Improvements

Here are some ideas to make the project better:

* Experiment with more algorithms: **Random Forests, Gradient Boosting (e.g. XGBoost, LightGBM), or even Neural Networks**.
* Feature engineering: add features like **GDP per capita**, **annual growth rates**, **income share ratios**, etc.
* Cross-validation / K-fold to better assess model stability.
* Handle outliers and missing data more carefully.
* Try dimensionality reduction (PCA, feature selection) if many features.
* Visualize predictions vs actuals with plots, error distributions, etc.
* Deploy the model (as a web API) so users can input feature values and get predicted Gini coefficient.

---

## 🗣️ How to Use

1. Ensure you have all feature inputs available: categorical & numerical.
2. Process categorical features (must match encoding used during training).
3. Load the trained model (`gini.pkl`).
4. Prepare a feature vector with the right order of features.
5. Call `model.predict(...)` to get the Gini coefficient.

Example (in Python):

```python
import pickle
# load model
with open('gini.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input
# [encoded_area, encoded_subarea, encoded_country, encoded_interpolated, encoded_region_wb,
#  year, population, gdp, p1, p2, p3, p4]
x = [[3, 13, 218, 0, 0, 1950, 2.536165e9, 4031.34, 0.00198, 0.006238, 0.008008, 0.01278]]
prediction = model.predict(x)
print(f"Predicted Gini Coefficient: {prediction[0]}")
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Author & Contributions

* **Author**: Palepu Hemasiri
* If you want to contribute: feel free to fork, open pull requests, suggest new features, report bugs, or send improvements.

---

## 🔗 References & Resources

* World Bank / UN data on Gini coefficients
* Scikit-learn documentation (regression, decision tree)
* Tutorials on feature engineering and model evaluation


