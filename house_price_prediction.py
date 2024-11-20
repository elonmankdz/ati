import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

df = pd.read_csv('./housing_dataset.csv')
X = df[["RM", "LSTAT", "PTRATIO"]]
y = df["MEDV"]

# Split the data into train and test datasets (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=80)

class HousePricePredictionModel:
    """
    A model predict price of house based on average number of rooms per dwelling (RM),
    lower status of the population (LSTAT) and pupil-teacher ratio by town (PTRATIO)
    """
    def __init__(self, X, y):
        self.X = np.c_[np.ones(X.shape[0]), X]  # Add intercept column
        self.y = y
        self.intercept: int | None = None
        self.coefficients: list | None = None

        # Compute X^T X
        X_transpose = self.X.T
        X_transpose_X = np.dot(X_transpose, self.X)

        # Compute (X^T X)^-1
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)

        # Compute X^T y
        X_transpose_y = np.dot(X_transpose, self.y)

        # Compute intercept and coefficients
        result = np.dot(X_transpose_X_inv, X_transpose_y)
        self.intercept = result[0]
        self.coefficients = result[1:]

        """
        Predicts the house price based on rm, lstats and ptratio

        Parameters:
        ----
        - rm (float): The average number of rooms per dwelling (RM).
        - lstat (float): The percentage of the population with lower socioeconomic status (LSTAT).
        - ptratio (float): The pupil-teacher ratio by town (PTRATIO).

        Returns:
        ----
        - float: the house price (MEDV).
        """

    def predict_price(self, rm, lstat, ptratio):
        return self.intercept + self.coefficients[0]*rm + self.coefficients[1]*lstat + self.coefficients[2]*ptratio

def draw_scatter_plot(x_data, y_data):
    # Plot the comparison between predicted and actual prices
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='blue', alpha=0.6,
                label='Predicted vs Actual')
    plt.plot([x_data.min(), x_data.max()], [x_data.min(), x_data.max()],
             color='red', lw=2, label='Ideal Prediction Line')
    plt.title('Comparison of Predicted vs Actual House Prices')
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_line_chart(x_data, y_data):
    # Plot the actual vs predicted prices using a Line Chart
    plt.plot(range(len(y_data)), y_data, label="Actual Prices", color='blue', marker='o', linestyle='-', linewidth=2)
    plt.plot(range(len(x_data)), x_data, label="Predicted Prices", color='red', marker='x', linestyle='--', linewidth=2)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Price ($)')
    plt.title('Actual vs Predicted Housing Prices')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

def predict():
    try:
        rm = float(entry_rm.get())
        lstat = float(entry_lstat.get())
        ptratio = float(entry_ptratio.get())
        price = model.predict_price(rm, lstat, ptratio)
        result_label.config(text=f"Predicted Price: ${price:.2f}", foreground="green")
    except ValueError:
        result_label.config(text="Please enter valid numerical values.", foreground="red")

def show_ui():
    # Create the main window
    window = tk.Tk()
    window.title("House Price Predictor")
    window.geometry("400x300")
    window.configure(bg="#f7f9fc")

    # Title Label
    title_label = tk.Label(window, text="House Price Predictor",
                           font=("Helvetica", 16, "bold"), bg="#f7f9fc",
                           fg="#333")
    title_label.pack(pady=10)

    # Create a frame for the inputs
    input_frame = tk.Frame(window, bg="#ffffff", padx=10, pady=10,
                           relief="ridge", bd=2)
    input_frame.pack(pady=10)

    # Input fields
    tk.Label(input_frame, text="Avg. Rooms per Dwelling (RM):",
             font=("Helvetica", 10), bg="#ffffff").grid(row=0, column=0,
                                                        sticky="w", pady=5)
    entry_rm = tk.Entry(input_frame, width=20)
    entry_rm.grid(row=0, column=1, pady=5)

    tk.Label(input_frame, text="Lower Status Population (LSTAT):",
             font=("Helvetica", 10), bg="#ffffff").grid(row=1, column=0,
                                                        sticky="w", pady=5)
    entry_lstat = tk.Entry(input_frame, width=20)
    entry_lstat.grid(row=1, column=1, pady=5)

    tk.Label(input_frame, text="Pupil-Teacher Ratio (PTRATIO):",
             font=("Helvetica", 10), bg="#ffffff").grid(row=2, column=0,
                                                        sticky="w", pady=5)
    entry_ptratio = tk.Entry(input_frame, width=20)
    entry_ptratio.grid(row=2, column=1, pady=5)

    # Predict button
    predict_button = tk.Button(window, text="Predict Price", command=predict)
    predict_button.pack(pady=10)

    # Result label (directly below the button)
    result_label = tk.Label(window, text="", font=("Helvetica", 12),
                            bg="#f7f9fc", fg="green")
    result_label.pack(pady=10)

    # Run the application
    window.mainloop()

if __name__ == '__main__':
    model = HousePricePredictionModel(X_train, y_train)

    # Make predictions on the test set
    predictions = [model.predict_price(row["RM"], row["LSTAT"], row["PTRATIO"]) for _, row in X_test.iterrows()]

    # Calculate Mean Squared Error (MSE)/R-squared score
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    draw_line_chart(y_test, predictions)
    draw_scatter_plot(y_test, predictions)

    show_ui()

