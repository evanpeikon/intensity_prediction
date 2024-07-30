# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load data
filename = 'O2.csv'
names = ['time', 'SmO2', 'SmO2_accel', 'SmO2_jerk', 'acceleration','jerk']
data = pd.read_csv(filename, names=names)

# calculate rolling averages for all metrics and add them to the dataset
window_size = 10
data['SmO2_rolling_mean'] = data['SmO2'].rolling(window=window_size).mean()
data['SmO2_accel_rolling_mean'] = data['SmO2_accel'].rolling(window=window_size).mean()
data['SmO2_jerk_rolling_mean'] = data['SmO2_jerk'].rolling(window=window_size).mean()
data['acceleration_rolling_mean'] = data['acceleration'].rolling(window=window_size).mean()
data['jerk_rolling_mean'] = data['jerk'].rolling(window=window_size).mean()

# backfill missing values for newly calculated rolling averages
data = data.fillna(method='bfill')

# select features to include/exclude from model
features = data.drop(['time', 'SmO2', 'SmO2_accel', 'SmO2_jerk', 'acceleration', 'jerk'], axis=1)

# scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# perform k-means clustering
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# define cluster labels and map clsuter numebrs to itensity phase labels
cluster_labels = {0: 'Recovery', 1: 'Light', 2: 'Moderate', 3: 'Heavy', 4:'Severe'}
data['Phase'] = data['Cluster'].map(cluster_labels)

# plot the scatter plot with the clusters
plt.figure(figsize=(20, 7))
scatter = plt.scatter(data['time'], data['SmO2_rolling_mean'], c=data['Cluster'], cmap='viridis')
legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=[cluster_labels[i] for i in range(len(cluster_labels))], title="Phase")
plt.gca().add_artist(legend1)
plt.xlabel('Time')
plt.ylabel('SmO2')
plt.title('Clusters of Exercise Phases')
plt.colorbar(label='Cluster')
plt.show()

# create an interactive plot using ploty
fig = px.scatter(data_frame=data, x='time', y='SmO2_rolling_mean', color='Phase', color_discrete_map=cluster_labels, title='Clusters of Exercise Phases', labels={'time': 'Time', 'SmO2_rolling_mean': 'SmO2'})
fig.update_layout(legend_title_text='Phase', coloraxis_colorbar_title='Cluster')
fig.show()
