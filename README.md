# CodeOp-Process-of-Solving-AnyML-Problem
Fisrt ML lesson
## Activity

1. Set up environment: install scikit-learn
2. Create a Github repository (in README.md format) to store the data science questions that you want to answer; this will serve as a cheatsheet for you!
3. [Read this article](https://towardsdatascience.com/how-to-master-scikit-learn-for-data-science-c29214ec25b0), and bring back questions about terms and concepts you do not understand.

## Activity Review

When students ask questions about the article assigned to read, answer their questions and place the concepts and terms within the flow of step-by-step guide or within the cheat sheet. The goal here is for students to map the whole picture of what ML is and its process.

## TERMS

Scikit-learn = SciPy Toolkits package --- Machine Learning models, utility functions, post model analysis, evaluation
SciPy
Scikit-image
utility functions
post model analysis
evaluation

## 1. Data representation in scikit-learn

 scikit data representation --> tabular dataset
 supervised Learning -> x and y
 unsupervised Learning -> only x
    * x are independent variable, either quantitative or qualitative (features)
    * y is the dependant variable, is the target?response variable (target)
    
## 2. Loading data from CSV files via Pandas
 
 We can read the datasets throyght pandas --- pd.read_csv()
    - So we have Pandas DataFrame
 Data Preprocessing & Feature Engineering (drop column, fill values, replace values-feature transformation, choose the columns for your dataset, ...)
 
## 3. Utility functions from scikit-learn
  3.1. Creating artificial datasets
  
you can create artificial datasets using scikit-learn

from sklearn.datasets import make_classification
X,Y = make_classification(n_samples=200, n_classes=2, n_features=2,...)

  3.2. Feature scaling
  
 As features may be of heterogeneous scales with several magnitude difference, it is therefore essential to perform feature scaling.
    * normalization (scaling features to a uniform range of 0 and 1)
        * normalize()
    * standardization (scaling features such that they have centered mean and unit variance)-- mean=0, sd=1
        * StandardScaler()
  
  3.3. Feature selection
 
"A common feature selection approach that I like to use is to simply discard features that have low variance as they provide minimal signa"

  3.4. Feature engineering
  
There are features that are not suitable for building models, we have to transfor string in integers or floats -- binary numerical form

Two common types of categorical features includes:

    * Nominal features — Categorical values of the feature has no logical order and are independent from one another. For instance, categorical values pertaining to cities such as Los Angeles, Irvine and Bangkok are nominal.
    * Ordinal features — Categorical valeus of the feature has a logical order and are related to one another. For instance, categorical values that follow a scale such as low, medium and high has a logical order.
    
- Pandas (get_dummies() function and map() method)
- scikit-learn (OneHotEncoder(), OrdinalEncoder(), LabelBinarizer(), LabelEncoder(), etc.).
 
  3.5. Imputing missing data
  
Users can use either the univariate or multivariate imputation method via the SimpleImputer() and IterativeImputer() functions from the sklearn.impute sub-module.
  
  3.6. Data splitting
  
A commonly used function would have to be data splitting for which we can separate the given input X and y variables as training and test subsets (X_train, y_train, X_test and y_test).

from sklearn.model.selection import train_test_split

  3.7. Creating a workflow using Pipeline
  
Pipeline() function to create a chain or sequence of tasks that are involved in the construction of machine learning models. (feature imputation, feature encoding and model training)

## 4. High-level overview of using scikit-learn

  4.1. Core steps for building and evaluating models
  
    1. Import  ----from sklearn impor EstimatorName. ---- y = f(X).
    2. Instantiate. ----- model=EstimatorName().  ----- model, clf or rf
    3. Fit  -----model.fit(x_training,y_training).   ------ model building or model training.
    4. Predict ----- model.predict(x_test).  --- x_test = new data
    5. score ----model.score(x_test, y_test).  --- it could be a regression
    
  4.2. Model interpretation. ??? check again
  
  Model interpretation. --- random forest
  rf.feature_importances_
    
  4.3. Hyperparameter tuning
  
  perform hyperparameter tuning which we can perform via the use of the GridSearchCV() function.
    1. Firstly, we will create an artificial dataset and perform data splitting, which will then serve as the data for which to build subsequent models.
    2. Secondly, we will now perform the actual hyperparameter tuning
    3. Finally, we can display the results from hyperparameter tuning in a visual representation.
  
  
## 5. Example machine learning workflow

    1. Read in the data as a Pandas DataFrame and display the first and last few rows. ---PANDAS
    2. Display the dataset dimension (how many rows and columns).   ---PANDAS
    3. Display the data types of the columns and summarize how many columns are categorical and numerical data types.---PANDAS
    4. Handle missing data. First check if there are missing data or not. If there are, make a decision to either drop the missing data or to fill the missing data. Also, be prepared to provide a reason justifying your decision. ---PANDAS
    5. Perform Exploratory data analysis. Use Pandas groupby function (on categorical variables) together with the aggregate function as well as creating plots to explore the data. ---PANDAS
    6. Assign independent variables to the X variable while assigning the dependent variable to the y variable ---PANDAS
    7. Perform data splitting (using scikit-learn) to separate it as the training set and the testing set by using the 80/20 split ratio (remember to set the random seed number). ---scikit-learn / Pandas / matplotlib
    8. Use the training set to build a machine learning model using the Random forest algorithm (using scikit-learn). ---scikit-learn / Pandas / matplotlib
    9. Perform hyperparameter tuning coupled with cross-validation via the use of the GridSearchCV() function.?? ---scikit-learn / Pandas / matplotlib
    10. Apply the trained model to make predictions on the test set via the predict() function. ---scikit-learn / Pandas / matplotlib
    11. Explain the obtained model performance metrics as a summary Table with the help of Pandas (I like to display the consolidated model performance as a DataFrame) or also display it as a visualization with the help of matplotlib. ---scikit-learn / Pandas / matplotlib
    12. Explain important features as identified by the Random forest model.  ---scikit-learn / Pandas / matplotlib


## 6. Resources for Scikit-learn

  * Documentation (scikit-learn)
  * cheat sheets (scikit-learn, Datacamp)
