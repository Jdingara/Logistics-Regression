# Logistics-Regression
Rock vs Mine Prediction using Sonar Data

### Rock vs Mine Prediction using Sonar Data

This project focuses on building a machine learning model to predict whether a given object is a rock or a mine based on sonar signals. Using a logistic regression approach, the goal was to achieve high accuracy in classifying objects from the sonar dataset.

#### Project Overview:

- **Objective**: To develop a predictive system capable of classifying sonar signals as either rocks or mines.
  
- **Data Source**: The sonar dataset consists of 208 samples, each described by 60 frequency-based attributes. Each sample represents the response from a sonar signal aimed at an object, with the goal of identifying if the object is a mine or a rock.

- **Tools and Libraries Used**:
  - **Google Collaboratory**: For developing and executing Python code.
  - **Pandas**: Used for loading and manipulating the sonar dataset.
  - **Scikit-learn**: Employed for machine learning tasks, including the implementation of the logistic regression model.

#### Key Workflow Steps:

1. **Data Loading and Exploration**:
   - The dataset was imported and stored in a pandas DataFrame.
   - Initial data exploration was done using functions like `data.describe()` to get a statistical overview, including mean, variance, and data distribution, to understand its properties.

2. **Data Splitting**:
   - The dataset was divided into **training** and **test** sets to ensure the model could generalize well to new, unseen data. The training set was used to train the logistic regression model, while the test set was reserved for evaluating the model’s performance.

3. **Model Implementation**:
   - A **Logistic Regression** model was selected for its effectiveness in binary classification problems like this one.
   - After training, the model was evaluated on the test set and achieved an **80% accuracy** in predicting whether the object was a rock or a mine.

4. **Model Evaluation**:
   - The model’s performance was measured using accuracy as the primary metric, with a focus on improving prediction results through data preprocessing and parameter tuning.

#### Conclusion:

This project demonstrated the ability to utilize sonar data for classification purposes using a logistic regression model. With proper preprocessing and splitting of data, the model was able to achieve an accuracy of 80%. Further improvements could be made by exploring other machine learning models or enhancing feature engineering techniques.

