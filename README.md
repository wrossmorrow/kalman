
# Kalman

A straightforward, object-oriented approach to small Kalman Filters on GPS (lat/lon) data. 

# Quick Start

Actually used as follows with a DataFrame having lat/lon columns (ordered increasing in time): 
```
from .gps_kalman import KalmanFilter_CV_LL
...
KF = KalmanFilter_CV_LL() # (Q=np.diag([0.01,0.01,0.02,0.02]), R=np.diag([0.001,0.001]))
kf = KF(gps_data)

fig = plt.figure()
plt.scatter(gps_data.lat.values, gps_data.lon.values, c='k', s=5)
sizes = ( 10 + 200 * (kf.err - kf.err.min())/(kf.err.max() - kf.err.min()) ).values.astype('float')
plt.scatter(kf.lat.values, kf.lon.values, s=sizes, color=None, edgecolors='k', alpha=0.2)
plt.title('measure lat/long only (map)')
plt.xlabel('lat')
plt.ylabel('long')
fig.set_size_inches(14,5)
```

![A super simple sample filter](https://github.com/wrossmorrow/kalman/blob/main/sample.png?raw=true)
