# E-Commerce Product Categorization

This project aims to automate the classification of products into relevant categories based on their features. Accurate categorization helps streamline product searches and improve customer experience in e-commerce platforms.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)

---

## Overview
This project focuses on building a product categorization model that takes product details as input and assigns the most relevant category to it. It addresses challenges such as:
- Large number of categories
- Variability in product descriptions
- Accurate classification using machine learning techniques.

---

## Dataset
The dataset used contains information of e-commerce products, including:
- Description
- Category labels

---

## Technologies Used
- **Python**: Core programming language  
- **scikit-learn**: For building and training machine learning models  
- **Pandas**: Data manipulation and analysis  
- **NumPy**: Numerical computations  
- **Jupyter Notebook**: For experimentation and visualization

---

## Model
A machine learning model (e.g., Logistic Regression, Random Forest, or XGBoost) is trained to classify products based on the given features. Feature engineering techniques such as TF-IDF or Count Vectorizer may also be used to transform text data into numerical form.

---

## Usage

Train the model:
Run the Jupyter notebook or script to train the model on the prepared dataset.
Use the trained model to predict categories for new products using app.py 
   ```bash
   python app.py
```

## Result

The trained model achieves a good level of accuracy in classifying products into relevant categories. Evaluation metrics such as accuracy, precision, recall, and F1-score are used to assess model performance.


