import matplotlib.pyplot as plt
import numpy as np

def visualize_data():
    # Generate some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    z = np.cos(x)
    
    # Create a line plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Sine Wave', color='blue')
    plt.plot(x, z, label='Cosine Wave', color='red', linestyle='--')
    plt.title('Sine and Cosine Waves')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='green', marker='o')
    plt.title('Scatter Plot of Sine Wave Data')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.show()

# Example usage:
# visualize_data()
