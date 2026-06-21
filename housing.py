import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')

print(df.info())

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)

df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True, dtype=int)

target_column = 'median_house_value'
X_df = df.drop(columns=[target_column])
y_df = df[target_column]

X = X_df.values
y = y_df.values.reshape(-1, 1)

np.random.seed(42) # Set seed for reproducibility
shuffled_indices = np.random.permutation(len(X))
split_index = int(len(X) * 0.8)

X_train_raw = X[shuffled_indices[:split_index]]
y_train = y[shuffled_indices[:split_index]]

X_test_raw = X[shuffled_indices[split_index:]]
y_test = y[shuffled_indices[split_index:]]

train_mean = np.mean(X_train_raw, axis=0)
train_std = np.std(X_train_raw, axis=0)

X_train_scaled = (X_train_raw - train_mean) / train_std
X_test_scaled = (X_test_raw - train_mean) / train_std


train_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
corr_matrix = train_df.corr()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]
            print(f"{feature_1} and {feature_2}: {corr_matrix.iloc[i, j]:.2f}")

cols_to_drop = ['total_bedrooms', 'population', 'households']
indices_to_drop = [list(X_df.columns).index(col) for col in cols_to_drop]

X_train_scaled = np.delete(X_train_scaled, indices_to_drop, axis=1)
X_test_scaled = np.delete(X_test_scaled, indices_to_drop, axis=1)

final_features = list(X_df.columns)
final_features = [f for i, f in enumerate(final_features) if i not in indices_to_drop]
print(final_features)

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        """Trains the model using Vectorized Batch Gradient Descent"""
        N = len(y)
        
        # BIAS TRICK: Add a column of 1s to X
        ones_column = np.ones((N, 1))
        X_padded = np.concatenate((ones_column, X), axis=1)
        
        # Initialize weights to zeros
        self.weights = np.zeros((X_padded.shape[1], 1))
        
        # Gradient Descent Loop
        for i in range(self.epochs):
            y_pred = X_padded.dot(self.weights)
            error = y_pred - y
            mse = (1/N) * np.sum(error ** 2)
            self.loss_history.append(mse)
            gradients = (2/N) * X_padded.T.dot(error)
            self.weights -= self.lr * gradients

    def predict(self, X):
        """Predicts prices for new data"""
        ones_column = np.ones((len(X), 1))
        X_padded = np.concatenate((ones_column, X), axis=1)
        return X_padded.dot(self.weights)

# Initialize and train
# We use a learning rate of 0.1 for 1000 epochs to ensure it converges
model = CustomLinearRegression(learning_rate=0.01, epochs=1000000)
print("Training started... (this might take a few seconds with 16,000 rows)")
model.fit(X_train_scaled, y_train)
print("Training complete!")

# Evaluate on the Test Set
predictions = model.predict(X_test_scaled)

# Calculate Metrics
mse = np.mean((y_test - predictions) ** 2)
rmse = np.sqrt(mse) # Root Mean Squared Error gives us the error in actual dollars
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - predictions) ** 2)
r2_score = 1 - (ss_residual / ss_total)

print("\n--- Model Results ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared Score: {r2_score:.4f}")

# Display the learned weights to see which features were most important
print("\n--- Learned Weights ---")
print(f"Intercept (Base Value): ${model.weights[0][0]:,.2f}")
for i, feature in enumerate(final_features):
    print(f"Weight for {feature}: ${model.weights[i+1][0]:,.2f}")

# Plot the loss curve
plt.plot(model.loss_history)
plt.yscale('log')
plt.xscale('log')
plt.title("Gradient Descent: Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()