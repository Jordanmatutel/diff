import numpy as np
import pandas as pd

# Creates the amount of random data
n = 100

x_train = np.random.rand(n)

# Calculate the difference between our data. Add one more value to have the same size of other variables.
diff = np.diff(x_train)
diff = np.append(diff, 0)

# Makes the differential regression approximation for the next period of X
coefficient = np.polyfit(diff, x_train, 2)

# Test the differential regression using another set of data.
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

# Export the data into one document CSV
data = pd.DataFrame(columns=["x", "Differential Regression"])
data["X"] = x_train
data["Differential Regression"] = next_x
data.to_csv("data.csv", index=False)
