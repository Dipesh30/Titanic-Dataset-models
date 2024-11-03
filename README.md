# Titanic Dataset Analysis and Prediction

## Overview

This repository contains a project focused on analyzing the Titanic dataset to understand the factors that influenced survival and to build a predictive model to estimate the number of lives saved during the Titanic disaster.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Analysis](#analysis)
- [Prediction Model](#prediction-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Titanic dataset is a well-known dataset used for data analysis and machine learning tasks. It contains information about the passengers, including their age, gender, class, and whether they survived the tragedy. This project aims to explore the dataset, perform statistical analysis, and create a predictive model to estimate survival based on various features.

## Dataset

The dataset can be obtained from the Kaggle Titanic competition page: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data). The main features in the dataset include:

- **PassengerId**: Unique ID for each passenger
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- **Survived**: Survival status (0 = No, 1 = Yes)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Analysis

In the analysis phase, the project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) to visualize survival rates by different factors (e.g., age, gender, class)
- Correlation analysis to understand relationships between features

The analysis results are documented in the Jupyter Notebooks located in the `notebooks/` directory.

## Prediction Model

The prediction model aims to estimate survival based on the features in the dataset. The following steps are included:

1. Data preprocessing (handling missing values, encoding categorical variables)
2. Splitting the dataset into training and testing sets
3. Training a machine learning model (e.g., Logistic Regression, Random Forest, etc.)
4. Evaluating model performance using metrics like accuracy, precision, and recall

You can find the model implementation in the `src/` directory.

## Usage

To run the analysis and predictions, follow these steps:

1. Clone the repository:
   ```bash
     git@github.com:Dipesh30/Titanic-Dataset-models.git
   ```
2. Navigate to the project directory:
   ```bash
   cd titanic-dataset-analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare your dataset and place it in the `data/` directory.
5. Run the analysis notebook or script:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
6. To run the prediction model, execute:
   ```bash
   python src/predict.py
   ```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

Feel free to adjust any sections to better reflect your project's specifics!
