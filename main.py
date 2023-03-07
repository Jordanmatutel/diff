import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creates the amount of random data
n = 100

x_train = np.random.rand(n)
x_test = np.random.rand(n)
noise_train = np.random.normal(0, 0.3, n)
noise_test = np.random.normal(0, 0.3, n)

# Calculate the difference between our data.
diff = np.diff(x_train)
diff = np.append(diff, 0)

diff_test = np.diff(x_test)
diff_test = np.append(diff_test, 0)

# Makes the differential regression approximation for the next period of X
coefficient = np.polyfit(noise_train, diff, 2)

# Test the differential regression using another set of data.
result_list = []
for i in range(len(diff_test)):
    w2 = coefficient[0]
    w = coefficient[1]
    m = coefficient[2]
    result = (w2 * diff_test[i]) + (w * diff_test[i]) + m
    result_list.append(result)

# Makes the difference between the X of the actual period and to the X of the next period.
next_x = []
for i in range(len(result_list)):
    c = result_list[i] + noise_train[i]
    next_x.append(c)

# Creates the conventional lineal regression.
conventional = np.polyfit(noise_train, x_train, 2)

# Test the conventional lineal regression.
result_y = []
for i in range(len(x_test)):
    w2 = conventional[0]
    w = conventional[1]
    m = conventional[2]
    y = (w2 * x_test[i]) + (w * x_test[i]) + m
    result_y.append(y)

# Test and calculate the mean and the sum of error of the conventional lineal regression using the test data.
test = []
for i in range(len(x_test)):
    d = result_y[i] - noise_test[i]
    test.append(d)

test_loss = sum(test)
test_mean = test_loss / len(test)


# Sort the data
next_x.append(0)
x_train = np.insert(x_train, 0, 0)
noise_train = np.insert(noise_train, 0, 0)
result_y = np.insert(result_y, 0, 0)

# Export the data into one document CSV
data = pd.DataFrame(columns=["x", "y", "Differential Regression", "Lineal Regression"])
data["x"] = x_train
data["y"] = noise_train
data["Differential Regression"] = next_x
data["Lineal Regression"] = result_y
data.to_csv("data.csv", index=False)
