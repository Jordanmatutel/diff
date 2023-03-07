import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creates the amount of random data
n = 100

x_train = np.random.rand(n)

# Calculate the difference between our data. Add one more value to have the same size of other variables.
diff = np.diff(x_train)
diff = np.append(diff, 0)

# Makes the differential regression approximation for the next period of X
coefficient = np.polyfit(diff, x_train, 2)

# Take measure of the results of our linear regression..
result_list = []
for i in range(len(x_train)):
    w2 = coefficient[0]
    w = coefficient[1]
    m = coefficient[2]
    result = (w2 * x_train[i]) + (w * x_train[i]) + m
    result_list.append(result)

# Sort the data
x_train = np.insert(x_train, 0, 0)

# Makes the difference between the X of the actual period and to the X of the next period.
next_x = []
for i in range(len(result_list)):
    c = result_list[i] + x_train[i]
    next_x.append(c)

next_x.append(0)

# Creates one dataframe to fill our graph and saves it as a csv

data = pd.DataFrame(columns=["x", "Differential Regression", "Test", "Test Results"])
data["x"] = x_train
data["Differential Regression"] = next_x
data.to_csv("data.csv", index=False)

# Set the data for our graphs.
fig, ax = plt.subplots()

# Graph details
ax.plot(data['Differential Regression'], color='blue')
ax.scatter(data.index, data['x'], color='red')
max_y1 = max(data['Differential Regression'].max(), data['x'].max())
ax.set_ylim(0, max_y1+1)
ax.legend(['Differential Regression', 'X'])

plt.show()
