# Masoud Pourghavam
# Student Number: 810601044
# Course: Artificial Intelligence
# University of Tehran



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# Read the dataset
df = pd.read_csv("breast_cancer.csv")

# Check for missing data
print("Missing Data:\n", df.isnull().sum())

print ("#########################################################")

# Detect outliers
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
axs = axs.flatten()
for i, col in enumerate(df.columns[:9]):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_title(col)
plt.tight_layout()
plt.show()


# Find the outliers in each column
for column in df.columns[:-1]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}: {list(outliers.index)}")


#########################################################
print ("#########################################################")


# Replace values in column 10
df["Class"] = df["Class"].replace({2: 0, 4: 1})

# Print the updated dataset
print(df.head())

#########################################################
print ("#########################################################")

# Scale the data in columns 1 to 9
scaler = MinMaxScaler(feature_range=(0, 1))
df.iloc[:, :9] = scaler.fit_transform(df.iloc[:, :9])

# Print the updated dataset
print(df.head())

#########################################################
print ("#########################################################")

# Create a box plot to visualize the distribution of the data
sns.boxplot(data=df.iloc[:,0:9])

# Create histograms for columns 1 to 9
data = df.iloc[:, :9]

# Plot histograms for each column
data.hist(bins=20, figsize=(10, 8))
plt.show()

#########################################################
print ("#########################################################")

# Split the dataset into two based on tumor type
benign = df[df['Class'] == 0]
malignant = df[df['Class'] == 1]

# Plot the histogram for column 10
plt.hist([benign['Class'], malignant['Class']], bins=2, color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Histogram of Tumor Type')
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.show()

#########################################################
print ("#########################################################")

# Stack the columns
stacked = df.iloc[:, :9].stack().reset_index(drop=True)

# Plot the histogram
stacked.hist(bins=20, figsize=(10, 8))
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

#########################################################
print ("#########################################################")

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#########################################################
print ("#########################################################")

# Split the dataset into training, validation, and testing sets
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)

# Print the number of samples in each set
print("Number of samples in the training set:", len(train))
print("Number of samples in the validation set:", len(val))
print("Number of samples in the testing set:", len(test))

#########################################################
print ("#########################################################")


# Split the data into input (X) and output (y) variables
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_val = val.iloc[:, :-1]
y_val = val.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Train a logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = lr.predict(X_val)

# Evaluate the accuracy of the model on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation accuracy:", accuracy)

#########################################################
print ("#########################################################")

# Split the data into input (X) and output (y) variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Create a logistic regression object
lr = LogisticRegression(random_state=42)

# Evaluate the model using k-fold cross-validation with k=5
scores = cross_val_score(lr, X, y, cv=5)

# Print the mean and standard deviation of the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

#########################################################
print ("#########################################################")


# Train the model on the training data
lr.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = lr.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Confusion matrix values
TP = 39
FP = 0
FN = 4
TN = 26

# Calculate accuracy
accuracy = (TP + TN) / (TP + FP + FN + TN)
print("Accuracy:", accuracy)

# Calculate error rate
error_rate = (FP + FN) / (TP + FP + FN + TN)
print("Error rate:", error_rate)

# Calculate precision
precision = TP / (TP + FP)
print("Precision:", precision)

# Calculate sensitivity (recall)
sensitivity = TP / (TP + FN)
print("Sensitivity (Recall):", sensitivity)

# Calculate specificity
specificity = TN / (TN + FP)
print("Specificity:", specificity)

#########################################################
print ("#########################################################")



