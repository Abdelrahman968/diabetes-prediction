
# Diabetes Prediction 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Diabetes Prediction is a machine learning project aimed at predicting whether a person has diabetes based on specific health parameters. The project leverages data-driven techniques to build a predictive model that can help in early diagnosis and intervention. The model uses various algorithms, including Random Forest, to achieve high accuracy in predictions.

---

## Features
- **Preprocessing of data**: Handling missing values and scaling features for better performance.
- **Feature engineering**: Selection and transformation of input features.
- **Machine learning models**: Implementation and evaluation of models such as Random Forest, Logistic Regression, and SVM.
- **Model serialization**: Saving the trained model for easy reuse and deployment.
- **User-friendly interface**: A script for testing the model with new data inputs.

---

## Technologies Used
- **Python 3.x**
- **Scikit-learn**: For model development and evaluation
- **Pandas**: For data manipulation
- **NumPy**: For numerical computations
- **Matplotlib** and **Seaborn**: For data visualization
- **Pickle**: For model serialization

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abdelrahman968/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Preprocess the dataset:
   Run the script to preprocess the data and split it into training and testing sets.

2. Run the model:
   ```bash
   run the Project-v2.ipynb
   ```

3. Classify new instances:
   Input the health parameters into the provided script to get    predictions.

---

## Dataset
The dataset used for training and testing the model includes various health features such as age, BMI, glucose levels, and medical history. An example dataset:

- [Diabetes prediction dataset from Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

Place the dataset in the `dataset-v2/` directory.

---

## Project Workflow
1. **Data Preprocessing:**
   - Handle missing values
   - Normalize and scale features
   - Split data into training and testing sets

2. **Feature Extraction:**
   - Select relevant features for better model performance

3. **Model Development:**
   - Train using algorithms like Random Forest, Logistic Regression, and SVM
   - Use hyperparameter tuning for optimal performance

4. **Evaluation:**
   - Generate confusion matrices and classification reports
   - Calculate metrics such as accuracy, precision, recall, and F1-score

---

## Results
The trained model achieved the following Accuracy results:
- Logistic Regression Model: 95.93%
- SVM Model: 95.91%
- Random Forest: 99.93%
- Gradient Boosting Classifier: 97.13%
- DecisionTree Model: 97.08%

---

## Future Enhancements
- Collect more diverse datasets to enhance model generalization.
- Implement deep learning models, such as neural networks, for potential improvement.
- Deploy the model using Flask, FastAPI, or other frameworks for a web application.
- Implement real-time data input for live prediction.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code adheres to the project's style guidelines and include relevant documentation.

---

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit).
