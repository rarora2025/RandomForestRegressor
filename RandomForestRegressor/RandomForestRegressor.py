import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#insert fileName w/ data here
df = pd.read_csv("data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')


#checking to make sure target exsists
if "estimated_stock_pct" not in df.columns:
    raise Exception("Target not present in the data")
    
X = df.drop(columns=["estimated_stock_pct"])
y = df["estimated_stock_pct"]

accuracy = []
K=10
SPLIT = 0.75

for fold in range(0, K):

    model = RandomForestRegressor()

    # Create training and test samples, training model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)
    trained_model = model.fit(X_train, y_train)

    # Gtest sample
    y_pred = trained_model.predict(X_test)

    # Compare accuracy, using mae
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    accuracy.append(mae)
    print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    
print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
