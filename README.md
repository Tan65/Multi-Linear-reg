# Multi-Linear-reg
Consider only the below columns and prepare a prediction model for predicting Price.  Corolla&lt;-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data

data = pd.read_csv("ToyotaCorolla.csv", encoding='latin1')

# Select relevant columns
selected_columns = ["Price", "Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]
data = data[selected_columns]

# Data preprocessing - handle missing values, one-hot encoding for categorical variables if needed

# Split the data into features (X) and target variable (y)
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Univariate Analysis: Histogram
data.hist(figsize=(15, 10), bins=20)
plt.suptitle("Histograms of Numeric Variables", fontsize=16)
plt.show()

# Bivariate Analysis: Scatter Plot
sns.pairplot(data)
plt.suptitle("Pairplot of Numeric Variables", fontsize=16)
plt.show()
# Multivariate Analysis: Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16)
plt.show()
