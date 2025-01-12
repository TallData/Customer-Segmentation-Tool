import pandas as pd

# Load the CSV files into DataFrames
customers = pd.read_csv('customer_demographics.csv')
purchase_history = pd.read_csv('purchase_history.csv')
behavior_data = pd.read_csv('customer_behavior.csv')

# Merge the tables based on 'CustomerID'
merged_data = customers.merge(purchase_history, on='CustomerID').merge(behavior_data, on='CustomerID')

# Basic summary analysis
summary = merged_data.describe(include='all')

# Display the summary analysis
print("Summary Analysis:")
print(summary)

# Save the merged data as a CSV file
merged_data.to_csv('merged_customer_data.csv', index=False)

print("Merged data saved as 'merged_customer_data.csv'.")
