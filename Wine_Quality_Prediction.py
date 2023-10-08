
#_______________________________PREDICTING WINE QUAILTY WITH LINEAR REGRESSION________________________________________


#Data analysis libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sbn

pd.set_option('display.width', 1200)
pd.set_option('display.max_column', None)

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


#Read in and Explore the Data.
df = pd.read_csv('winequality-red.csv',delimiter=";")
print(df.describe())


#Data Analysis
#get a list of the features within the dataset
print("\n\ndf: \n",df.columns)

#From above output we find that there are 12 features in our dataset


#See a sample of the dataset to get an idea of the variables
print("\n\nFirst 5 rows of dataset : ")
print(df.head())


#Check types of features
print("\n\nData types for each feature :")
print(df.dtypes)
#Data types for each feature :
#fixed acidity           float64
#volatile acidity        float64
#citric acid             float64
#residual sugar          float64
#chlorides               float64
#free sulfur dioxide     float64
#total sulfur dioxide    float64
#density                 float64
#pH                      float64
#sulphates               float64
#alcohol                 float64
#quality                   int64
#dtype: object

#The all features are of float or int data type
#So, we have no need to to transform some of the data,
#such as converting categorical variables to numerical values.


#See a summary of the dataset
print("\n\n",df.describe(include="all"))

#There are a total of 1599  in our dataset.


#check for all the missing values in the dataset
print("\n\nMissing Values:")
print(pd.isnull(df).sum())
#Missing Values:
#fixed acidity           0
#volatile acidity        0
#citric acid             0
#residual sugar          0
#chlorides               0
#free sulfur dioxide     0
#total sulfur dioxide    0
#density                 0
#pH                      0
#sulphates               0
#alcohol                 0
#dtype: int64

#From above output we find that there are no missing values.
#So there is no need of cleaning the data.


# Exploratory Data Analysis (EDA)
# Explore the distribution of wine quality scores
# Question 1: What is the distribution of the wine quality scores?
plt.hist(df['quality'])
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Scores')
plt.show()





# Question 2: What are the relationships between the different features?
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation between Features')
plt.show()



# Question 3: Are there any outliers in the data?
# Visualize box plots for each feature
plt.figure(figsize=(12, 8))
df.boxplot(column=list(df.columns[:-1]), grid=False)
plt.title('Box Plots of Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()




# Model Building
# Split the data into training and test sets
X = df.drop("quality", axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit a linear regression model04

model = LinearRegression()
model.fit(X_train, y_train)


# Model Evaluation

# Question 4: What is the accuracy of the linear regression model?
accuracy = model.score(X_test, y_test)
print(f"\nAccuracy of the linear regression model: {accuracy}")
#Accuracy of the linear regression model: 0.4031803412796232



# Question 5: What are the most important features for the linear regression model?
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.coef_}).sort_values('importance', ascending=False)
print("\nMost important features for the linear regression model:")
print(feature_importances)
#Most important features for the linear regression model:
#                 feature  importance
#9              sulphates    0.841172
#10               alcohol    0.281890
#0          fixed acidity    0.023085
#3         residual sugar    0.006564
#5    free sulfur dioxide    0.005627
#6   total sulfur dioxide   -0.003644
#2            citric acid   -0.140821
#8                     pH   -0.393688
#1       volatile acidity   -1.001304
#4              chlorides   -1.806503
#7                density  -10.351594



# Question 6: What is the MSE of the linear regression model?
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE of the linear regression model: {mse}")
#MSE of the linear regression model: 0.39002514396395405



# Calculate root mean squared error (RMSE)
rmse = np.sqrt(mse)
print("\nRoot Mean Squared Error (RMSE):", rmse)
#Root Mean Squared Error (RMSE): 0.6245199307980123



# Question 7: What is the R-squared of the linear regression model?
r2 = r2_score(y_test, y_pred)
print(f"\nR-squared of the linear regression model: {r2}")
#R-squared of the linear regression model: 0.4031803412796232



# Step 8: How can you improve the performance of the linear regression model?
# In order to improve the performance of the linear regression model, you can:
# -> Perform feature selection to include only the most relevant features.
# ->Apply data normalization or standardization to ensure that all features are on a similar scale.
# -> Explore other regression algorithms and compare their performance.



# Step 9: What are the limitations of the linear regression model?
# The limitations of the linear regression model include:
# -> Linearity assumption: Linear regression assumes a linear relationship between the features and
#                         the target variable.
# -> Sensitivity to outliers: Linear regression can be sensitive to outliers, which can affect the
#                            model's performance.
# -> Independence of features: Linear regression assumes that the features are independent of each
#                             other, which may not always hold true.
# -> Normality assumption: Linear regression assumes that the residuals are normally distributed.



# Step 10: What are the implications of your findings for the real-world problem?
# Based on the findings, we can predict wine quality using the linear regression model and
# identify the most important features that influence wine quality.This information can be
# valuable for winemakers to understand the factors affecting wine quality and make informed
# decisions in the production process.
