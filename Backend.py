import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the cleaned dataset
file_path = r"C:\Users\kamal\Downloads\walmart Retail Dataset.xlsx"
df = pd.ExcelFile(file_path).parse('walmart Retail Data')

# Feature Engineering: Encode categorical variables
df['Order Month'] = df['Order Date'].dt.to_period('M')
df['Category_Code'] = df['Product Category'].astype('category').cat.codes

# Prepare features and target
X = df[['Category_Code', 'Customer Age', 'Product Base Margin', 'Sales']]
y = df['Profit']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'sales_prediction_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
