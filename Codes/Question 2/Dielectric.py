# Masoud Pourghavam
# Student Number: 810601044
# Course: Artificial Intelligence
# University of Tehran


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



# read the dataset into a pandas DataFrame
df = pd.read_excel('Performance-Degradation Data Nelson.xlsx')

# create a linear regression object
reg = LinearRegression()

# fit the model using the x1 and x2 columns as features and y column as target variable
reg.fit(df[['x1', 'x2']], df['y'])


# create a meshgrid of x1 and x2 values
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 100)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# create a feature matrix using the meshgrid
X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))

# predict the target variable for the feature matrix
y_mesh = reg.predict(X_mesh)

# plot the dataset and the regression line
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'], c='b', marker='o')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh.reshape(x1_mesh.shape), alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()


######################################################################
print("############################################")


# create an SVR object with RBF kernel
reg = SVR(kernel='rbf')

# fit the model using the x1 and x2 columns as features and y column as target variable
reg.fit(df[['x1', 'x2']], df['y'])
# create a meshgrid of x1 and x2 values
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 100)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# create a feature matrix using the meshgrid
X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))

# predict the target variable for the feature matrix
y_mesh = reg.predict(X_mesh)

# plot the dataset and the regression surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'], c='b', marker='o')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh.reshape(x1_mesh.shape), alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

######################################################################
print("############################################")

# create an SVR object with 2nd order polynomial kernel
reg = SVR(kernel='poly', degree=2)

# fit the model using the x1 and x2 columns as features and y column as target variable
reg.fit(df[['x1', 'x2']], df['y'])
# create a meshgrid of x1 and x2 values
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 100)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# create a feature matrix using the meshgrid
X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))

# predict the target variable for the feature matrix
y_mesh = reg.predict(X_mesh)

# plot the dataset and the regression surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'], c='b', marker='o')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh.reshape(x1_mesh.shape), alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

######################################################################
print("############################################")

# create an SVR object with sigmoid kernel
reg = SVR(kernel='sigmoid')

# fit the model using the x1 and x2 columns as features and y column as target variable
reg.fit(df[['x1', 'x2']], df['y'])
# create a meshgrid of x1 and x2 values
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 100)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# create a feature matrix using the meshgrid
X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))

# predict the target variable for the feature matrix
y_mesh = reg.predict(X_mesh)

# plot the dataset and the regression surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'], c='b', marker='o')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh.reshape(x1_mesh.shape), alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

######################################################################
print("############################################")



# define the dataset as df
X = df[['x1', 'x2']]
y = df['y']

# define the k-fold cross-validation object
kf = KFold(n_splits=4, shuffle=True)

# define a dictionary to store the results
results = {}

# iterate over each kernel and train the model using k-fold cross-validation
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    mae_scores = []
    r2_scores = []
    for train_idx, test_idx in kf.split(X):
        # split the dataset into train and test sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # create the regression model
        reg = SVR(kernel=kernel)

        # train the model on the training set
        reg.fit(X_train, y_train)

        # make predictions on the test set
        y_pred = reg.predict(X_test)

        # calculate the mean absolute error and R2-score
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # store the mean MAE and R2-score for the current kernel in the dictionary
    results[kernel] = {'mean_mae': np.mean(mae_scores), 'mean_r2': np.mean(r2_scores)}

# print the results
for kernel in results:
    print(f"Kernel: {kernel}")
    print(f"Mean MAE: {results[kernel]['mean_mae']:.4f}")
    print(f"Mean R2-score: {results[kernel]['mean_r2']:.4f}")

######################################################################
print("############################################")




# define the dataset as df
X = df[['x1', 'x2']]
y = df['y']

# create a dictionary to store the kernels
kernels = {'linear': 'linear',
           'rbf': 'rbf',
           'poly': 'poly',
           'sigmoid': 'sigmoid'}

# iterate over each kernel and each value of alpha to create the corresponding regression model
for kernel_name, kernel in kernels.items():
    for alpha in [1, 2]:
        # create the Kernel Ridge regression model with L2 regularization
        reg = KernelRidge(alpha=alpha, kernel=kernel)

        # compute the mean absolute error and R2-score using k-fold cross-validation
        mae = -cross_val_score(reg, X, y, cv=4, scoring='neg_mean_absolute_error').mean()
        r2 = cross_val_score(reg, X, y, cv=4, scoring='r2').mean()

        # print the results
        print(f"Kernel: {kernel_name}, Alpha: {alpha}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2-score: {r2}")

######################################################################
print("############################################")
print("Iinear Kernel:")



# Split the dataset into features and target
X = df[['x1', 'x2']]
y = df['y']

# Define the regularization parameters to test
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Train a support vector regression model with a linear kernel for each regularization parameter
for param in reg_params:
    svr = SVR(kernel='linear', C=param)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2-score for regularization parameter {param}: {r2}")

# Use grid search to choose the best regularization parameter
reg_param_grid = {'C': reg_params}
svr_grid = GridSearchCV(SVR(kernel='linear'), reg_param_grid, scoring='r2')
svr_grid.fit(X, y)

best_param = svr_grid.best_params_['C']
best_r2 = svr_grid.best_score_

print(f"Best regularization parameter: {best_param}")
print(f"Best R2-score: {best_r2}")


######################################################################
print("############################################")
print("2nd order poly Kernel:")

# Define the regularization parameters to test
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Train a support vector regression model with a 2nd order polynomial kernel for each regularization parameter
for param in reg_params:
    svr = SVR(kernel='poly', degree=2, C=param)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2-score for regularization parameter {param}: {r2}")

# Use grid search to choose the best regularization parameter
reg_param_grid = {'C': reg_params}
svr_grid = GridSearchCV(SVR(kernel='poly', degree=2), reg_param_grid, scoring='r2')
svr_grid.fit(X, y)

best_param = svr_grid.best_params_['C']
best_r2 = svr_grid.best_score_

print(f"Best regularization parameter: {best_param}")
print(f"Best R2-score: {best_r2}")

######################################################################
print("############################################")
print("3rd order poly Kernel:")

# Define the regularization parameters to test
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Train a support vector regression model with a 3rd order polynomial kernel for each regularization parameter
for param in reg_params:
    svr = SVR(kernel='poly', degree=3, C=param)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2-score for regularization parameter {param}: {r2}")

# Use grid search to choose the best regularization parameter
reg_param_grid = {'C': reg_params}
svr_grid = GridSearchCV(SVR(kernel='poly', degree=3), reg_param_grid, scoring='r2')
svr_grid.fit(X, y)

best_param = svr_grid.best_params_['C']
best_r2 = svr_grid.best_score_

print(f"Best regularization parameter: {best_param}")
print(f"Best R2-score: {best_r2}")

######################################################################
print("############################################")
print("4th order poly Kernel:")

# Define the regularization parameters to test
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Train a support vector regression model with a 4th order polynomial kernel for each regularization parameter
for param in reg_params:
    svr = SVR(kernel='poly', degree=4, C=param)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2-score for regularization parameter {param}: {r2}")

# Use grid search to choose the best regularization parameter
reg_param_grid = {'C': reg_params}
svr_grid = GridSearchCV(SVR(kernel='poly', degree=4), reg_param_grid, scoring='r2')
svr_grid.fit(X, y)

best_param = svr_grid.best_params_['C']
best_r2 = svr_grid.best_score_

print(f"Best regularization parameter: {best_param}")
print(f"Best R2-score: {best_r2}")

######################################################################
print("############################################")
print("RBF Kernel:")

# Define the regularization parameters to test
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Train a support vector regression model with an RBF kernel for each regularization parameter
for param in reg_params:
    svr = SVR(kernel='rbf', C=param)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2-score for regularization parameter {param}: {r2}")

# Use grid search to choose the best regularization parameter
reg_param_grid = {'C': reg_params}
svr_grid = GridSearchCV(SVR(kernel='rbf'), reg_param_grid, scoring='r2')
svr_grid.fit(X, y)

best_param = svr_grid.best_params_['C']
best_r2 = svr_grid.best_score_

print(f"Best regularization parameter: {best_param}")
print(f"Best R2-score: {best_r2}")


######################################################################
print("############################################")


# Define the kernels and regularization parameters to test
kernels = ['linear', 'poly2', 'poly3', 'poly4', 'rbf']
reg_params = [0.2, 0.8, 1, 5, 10, 20, 50, 300]

# Define the parameter grid for GridSearchCV
param_grid = [{'kernel': ['linear'], 'C': reg_params},
              {'kernel': ['poly'], 'C': reg_params, 'degree': [2, 3, 4]},
              {'kernel': ['rbf'], 'C': reg_params, 'gamma': ['scale', 'auto']}]

# Create a support vector regression model and perform grid search
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

# Print the best kernel and regularization parameter based on the highest cross-validation score
print(f"Best kernel: {grid_search.best_params_['kernel']} with regularization parameter C={grid_search.best_params_['C']}")