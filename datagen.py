import pandas as pd
import random
import string
from datetime import datetime, timedelta

# Create a function to generate random customer IDs
def generate_customer_id():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

# Generate customer demographics data
n_customers = 1000
customers = pd.DataFrame({
    'CustomerID': [generate_customer_id() for _ in range(n_customers)],
    'Age': [random.randint(18, 70) for _ in range(n_customers)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(n_customers)],
    'Income': [random.randint(20000, 100000) for _ in range(n_customers)],
    'Location': [random.choice(['Urban', 'Suburban', 'Rural']) for _ in range(n_customers)]
})

# Generate purchase history data with dates in the past
purchase_history = pd.DataFrame({
    'CustomerID': [random.choice(customers['CustomerID']) for _ in range(n_customers * 2)],
    'PurchaseDate': [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(n_customers * 2)],
    'AmountSpent': [round(random.uniform(10, 500), 2) for _ in range(n_customers * 2)]
})

# Generate behavior data with dates in the past
behavior_data = pd.DataFrame({
    'CustomerID': [random.choice(customers['CustomerID']) for _ in range(n_customers)],
    'PageViews': [random.randint(1, 100) for _ in range(n_customers)],
    'EmailSubscribed': [random.choice([True, False]) for _ in range(n_customers)],
    'LastLoginDate': [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(n_customers)]
})

# Save dataframes to CSV files
customers.to_csv('customer_demographics.csv', index=False)
purchase_history.to_csv('purchase_history.csv', index=False)
behavior_data.to_csv('customer_behavior.csv', index=False)

print("Sample data generated and saved as CSV files.")
