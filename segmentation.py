import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# Assuming you have loaded the merged data as 'merged_data'
merged_data = pd.read_csv('merged_customer_data.csv')

# # ---------- RFM Segmentation ----------
# # Calculate Recency, Frequency, and Monetary Value
# current_date = pd.to_datetime('2023-08-01')

# # Calculate the most recent purchase date for each customer
# recency_data = merged_data.groupby('CustomerID')['PurchaseDate'].max().reset_index()
# recency_data['RecencyDate'] = pd.to_datetime(recency_data['PurchaseDate'])
# recency_data['Recency'] = (current_date - recency_data['RecencyDate']).dt.days
# #recency_data.to_csv('merged_customer_test.csv', index=False)

# # Merge the recency data back into the merged_data DataFrame
# merged_data = pd.merge(merged_data, recency_data, on='CustomerID')

# merged_data.to_csv('merged_customer_test.csv', index=False)

# # Calculate Recency based on the most recent purchase date
# # merged_data['Recency'] = (current_date - merged_data['LastPurchaseDate']).dt.days

# rfm_df = merged_data.groupby('CustomerID').agg({
#     'Recency': 'min',
#     'CustomerID': 'count',
#     'AmountSpent': 'sum'
# }).rename(columns={
#     'CustomerID': 'Frequency',
#     'AmountSpent': 'Monetary'
# }).reset_index()

# # Standardize the RFM variables
# scaler = StandardScaler()
# rfm_df_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
# rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# # Apply K-Means clustering for RFM Segmentation
# kmeans_rfm = KMeans(n_clusters=4, random_state=0)
# rfm_df['RFM_Cluster'] = kmeans_rfm.fit_predict(rfm_df_scaled)


# # Visualize RFM Clusters
# sns.set(style="whitegrid")
# plt.figure(figsize=(8, 4))
# sns.scatterplot(x="Recency", y="Monetary", hue="RFM_Cluster", data=rfm_df, palette="viridis")
# plt.xlabel('Recency')
# plt.ylabel('Monetary')
# plt.title('RFM Segmentation')
# plt.savefig('rfm_segmentation.png')
# plt.close()

# # ---------- Behavioral Segmentation ----------
# # Hypothesis test to determine the PageViews cutoff
# page_views_above = merged_data[merged_data['PageViews'] >= 50]['AmountSpent']
# page_views_below = merged_data[merged_data['PageViews'] < 50]['AmountSpent']

# t_stat, p_value = ttest_ind(page_views_above, page_views_below)

# confidence_level = 0.99
# alpha = 1 - confidence_level

# if p_value < alpha:
#     optimal_cutoff = 50  # Significant difference found at 99% confidence
# else:
#     optimal_cutoff = 0

# print(f"Optimal PageViews Cutoff: {optimal_cutoff}")

# # Apply the optimal cutoff to Behavioral Segmentation
# merged_data['Behavioral_Segment'] = (merged_data['PageViews'] >= optimal_cutoff).map({True: 'Engaged', False: 'Inactive'})

# # Visualize the t-test results
# sns.set(style="whitegrid")
# plt.figure(figsize=(8, 4))
# sns.boxplot(x="Behavioral_Segment", y="AmountSpent", data=merged_data)
# plt.xlabel('Behavioral Segment')
# plt.ylabel('AmountSpent')
# plt.title('Behavioral Segmentation - PageViews vs. AmountSpent')
# plt.savefig('behavioral_segmentation.png')
# plt.close()

# ---------- Machine Learning-Based Clustering ----------
# Assuming you have prepared your data and scaled it as 'X_scaled'

# Select the columns to be scaled (e.g., Age, Income, PageViews, AmountSpent)
columns_to_scale = ['Age', 'Income', 'PageViews', 'AmountSpent']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
Scaled = scaler.fit_transform(merged_data[columns_to_scale])

X_scaled = pd.DataFrame(Scaled)

# Save the scaled data
X_scaled.to_csv('scaled_merged_customer_data.csv', index=False)

# You can use the Silhouette Score to determine the number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Determine the optimal number of clusters based on the highest Silhouette Score
optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 due to range starting from 2

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=0)
merged_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Perform Silhouette Analysis for cluster personas
silhouette_avg = silhouette_score(X_scaled, merged_data['Cluster'])

sample_silhouette_values = silhouette_samples(X_scaled, merged_data['Cluster'])
merged_data['Silhouette_Score'] = sample_silhouette_values

# Visualize the silhouette analysis using PCA-transformed data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colors = plt.cm.tab20(merged_data['Cluster'] / optimal_n_clusters)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Silhouette Analysis - Avg Score: {silhouette_avg:.2f}')
plt.savefig('silhouette_analysis.png')
plt.close()

# ---------- Customer Personas ----------
# Define customer personas based on cluster characteristics (4 personas)
personas = {
    0: "High-Income Shoppers",
    1: "Young Professionals",
    2: "Frequent Shoppers",
    3: "Value-Conscious Consumers"
}

# Output customer personas with Silhouette Scores
for cluster, persona in personas.items():
    cluster_data = merged_data[merged_data['Cluster'] == cluster]
    avg_silhouette_score = cluster_data['Silhouette_Score'].mean()
    
    print(f"**Persona: {persona}**")
    print(f"Total Customers: {len(cluster_data)}")
    print(f"Average Age: {cluster_data['Age'].mean():.2f} years")
    print(f"Average Income: ${cluster_data['Income'].mean():,.2f}")
    print(f"Average PageViews: {cluster_data['PageViews'].mean():.2f}")
    print(f"Average AmountSpent: ${cluster_data['AmountSpent'].mean():,.2f}")
    print(f"Average Silhouette Score: {avg_silhouette_score:.2f}\n")

# Save the personas visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)  # Use X_pca here
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cluster Personas - Color Coded by Persona')
plt.savefig('cluster_personas.png')
plt.close()