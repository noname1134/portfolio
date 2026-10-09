import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([
    [20, 20000],
    [30, 50000],
    [40, 80000],
    [50, 110000],
    [60, 140000]
])

scaler = StandardScaler()

scaled = scaler.fit_transform(data)

print("Original:")
print(data)

print("\nScaled:")
print(scaled)

print("\nMeans:")
print(scaled.mean(axis=0))

print("\nStandard deviations:")
print(scaled.std(axis=0))