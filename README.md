# 🚴‍♂️ Project Introduction (Scientific Background)
### Muscle Oxygenation Reflects The Bodies Response To Exercise
Muscle oxygenation (SmO2) is the percentage of blood that is oxygenated in skeletal muscle. People often think of muscle oxygenation (SmO2) as a measure of exercise intensity, and while SmO2 levels are associated with exercise intensity, they are not the same.

A better way to think about muscle oxygenation is that it measures our body's response to exercise intensity. This delineation is critical and is why devices like the NNOXX wearable can be used to quantify the physiological effects of exercise. 

To understand why SmO2 is not a measure of intensity itself, we must appreciate that muscle oxygenation reflects the balance of oxygen supply and demand in exercising muscles. Under most circumstances, increasing exercise intensity disproportionately increases the muscle's demand for oxygen relative to supply, and thus, muscle oxygenation decreases. However, this isn't always the case — there are times when we increase exercise intensity and muscle oxygenation remains stable.

Understanding that muscle oxygenation represents our body's response to exercise intensity, not intensity itself, opens up a new way of interpreting our SmO2 measurements. 

### Muscle Oxygenation and It’s First and Second Derivatives:
In addition to looking at SmO2, there is value in exploring it’s rate of change, which is referred to as ΔSmO2, or SmO2'. The relationship between SmO2 and SmO2' is analogous to the relationship between velocity and acceleration. Whereas SmO2 represents your oxygenation saturation at a given point in time, SmO2' represents the rate that it changes over time. 

Thus, a positive rate of change (i.e., SmO2' >0) means SmO2 is increasing and oxygen supply supersedes oxygen utilization. Generally, this occurs during the recovery phase after intense exercise when the body is replenishing oxygen stores. On the other hand, a negative rate of change (i.e., SmO2' <0) means SmO2 is decreasing and oxygen utilization is outstripping oxygen supply. As a result, we tend to see negative SmO2' values during high intensity exercise. Finally, a SmO2' of ~0%/sec means that SmO2 levels are unchanging and that a metabolic steady state is occurring.

Now, In addition to calculation your SmO2', you can also calculate your SmO2'', which is the rate of change of SmO2' (SmO2'' is analogous to jerk, which is the rate of change of acceleration). If SmO2 is decreasing (i.e., SmO2' is negative), but SmO2'' is positive, it indicates that the decline in muscle oxygenation is slowing down. In other words, while oxygenation levels are still dropping, they are doing so at a slower rate. Alternatively, if SmO2 is increasing (i.e., SmO2' is positive) and SmO2'' is positive, it indicates that the muscle oxygenation levels are increasing at an accelerating rate.

On the other hand, is SmO2 is decreasing (i.e., SmO2' is negative) and SmO2'' is negative, it indicates that the decline in muscle oxygenation is accelerating. In other words, oxygenation levels are dropping more rapidly. Additionally, if SmO2 is increasing (i.e., SmO2' is positive) but SmO2'' is negative, it indicates that the increase in muscle oxygenation is slowing down. While oxygenation levels are still rising, they are doing so at a slower rate.

As a result, the combination of SmO2, SmO2', and SmO2'' can be used to understand physiological mechanisms, such as the dynamics of oxygen transport and utilization, vascular responses, and metabolic changes during different exercise phases. Additionally, these metrics can be used to detect transition phases and critical physiological thresholds, which i’ll be exploring in the remainder of this post. 

# 🚴‍♂️ Project Walkthrough 

In the sections above, we've explored how muscle oxygenation measurement and its first and second derivatives can be used to better understand the body's physiological response to exercise. However, working with biological time series data of this nature is incredibly complex. As a result, it's extremely difficult for a human practitioner to look at their data and discern where phase transitions between different exercise intensities occur.

To address this challenge, I aimed to develop an unsupervised machine learning model capable of predicting moment-to-moment exercise intensity using an individual's biomarker and movement data. By unsupervised model, I mean that we are not providing our algorithm with any information about the user's exertion level or relative intensity during exercise. The biomarker and movement data referenced include muscle oxygenation (SmO2) and its derivatives, as well as acceleration and its derivatives, all of which can be measured with the NNOXX wearable device. Below, I'll walk you through a model created using data from an athlete performing a ramp incremental exercise test:

```python
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
```
Which produces the following output:



To generate these exercise intensity classifications/predictions, my model was tuned to include five distinct clusters using rolling average of SmO2, SmO2', SmO2'', acceleration, and jerk as inputs. I then manually labeled the clusters based on a standard five-zone intensity distribution model commonly used in endurance sports. Interestingly, the predictions matched my expectations perfectly. The dark purple sections of this athlete's muscle oxygenation trend labeled “Recovery” indeed corresponded to the momentary recovery periods between load steps during the ramp incremental exercise test. Additionally, you can see that at the start of each work bout, the athlete quickly transitions through different intensity zones before vacillating between the moderate to severe zones in varying proportions, depending on the specific load step.

Additionally, using the code below I was able to create an interactive visualization, which allows me zoom into the chart, pan around, and more. By doing so, we can observe phase transitions between intensity zones on very short time scales. You can use the code below, and the data stored in the 'data' folder of this project repository to create your own interactive visualization to view this data:
```
fig = px.scatter(data_frame=data, x='time', y='SmO2_rolling_mean', color='Phase', color_discrete_map=cluster_labels, title='Clusters of Exercise Phases', labels={'time': 'Time', 'SmO2_rolling_mean': 'SmO2'})
fig.update_layout(legend_title_text='Phase', coloraxis_colorbar_title='Cluster')
fig.show()
```
Which produces the following output (still image of an interactive visualization tool:




To this point, it hasn’t been possible to witness the phase transitions between exercise intensity domains on such short time scales, which is a major limitations of zoning training based on physiological metrics such as heart rate and blood lactate. This highlights an advantage of using NIRS-based indicators of exercise response, like muscle oxygenation, which reflect local muscle physiological responses, compared to metrics like heart rate and lactate that are systemic indicators of exercise responses.

Additionally, this approach to predicting exercise intensity has advantages over traditional power-based zones, which are absolute indicators of intensity. Under normal circumstances, we should expect these two metrics to align. For example, if 350 watts corresponds to a heavy intensity domain (using critical power-based zoning), we should expect this model to indicate that the local muscle is experiencing heavy intensity. However, power-based zones established under ideal conditions at sea level may not be accurate when an athlete is heavily fatigued, at altitude, or in other varying conditions. This type of model can effectively reveal the difference between how we expect a given power output to stress the rider's muscles and how it is actually stressing their muscles.
