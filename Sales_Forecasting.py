import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('sales_data.csv')  # CSV with date, product, quantity, revenue

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

# Aggregating by date
sales = df.groupby('date')['revenue'].sum().reset_index()

# Feature Engineering
sales['days'] = (sales['date'] - sales['date'].min()).dt.days
X = sales[['days']]
y = sales['revenue']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Plot
plt.plot(sales['date'], sales['revenue'], label='Actual Sales')
plt.plot(sales['date'].iloc[-len(y_pred):], y_pred, label='Predicted Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Sales Forecasting with Linear Regression')
plt.legend()
plt.show()
