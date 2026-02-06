from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ayx import Alteryx


def simple_linear_regression():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    
    # Use only one feature for simplicity
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=0)
    
    # Create a linear regression object
    regr = LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    # Plot outputs
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Feature (e.g., BMI - standardized)')
    plt.ylabel('Disease Progression')
    plt.title('Linear Regression on Diabetes Dataset')
    #plt.show()
    
    output_path = r"C:\Temp\example_plot.png"
    plt.savefig(output_path)
    plt.close()
    
    # The coefficient
    coefficient = regr.coef_[0]
    print(f'Coefficient: {regr.coef_[0]:.2f}')
    # The mean squared error
    mse = np.mean((y_pred - y_test) ** 2)
    print(f'Mean squared error: {mse:.2f}')
    
    # Create a single-row DataFrame
    out_df = pd.DataFrame({
        "coefficient": [coefficient],
        "mean_squared_error": [mse]
    })

    # Write to output anchor 1
    Alteryx.write(out_df, 1)
