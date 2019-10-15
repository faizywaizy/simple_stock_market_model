# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:43:33 2019

To build a model to use volume to predict opening stock prices from closing
+ or -

@author: Faizan
"""

#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#make life easier by storing fil locations in variables
GAS_TRAIN_DATA = "CSV_Files/Gas Data Last Year.csv"
GAS_TEST_DATA = "CSV_Files/Gas Data Last Month.csv"

GOLD_TRAIN_DATA = "CSV_Files/Gold Data Last Year.csv"
GOLD_TEST_DATA = "CSV_Files/Gold Data Last Month.csv"

Oil_TRAIN_DATA = "CSV_Files/Oil Data Last Year.csv"
Oil_TEST_DATA = "CSV_Files/Oil Data Last Month.csv"

SILVER_TRAIN_DATA = "CSV_Files/Silver Data Last Year.csv"
SILVER_TEST_DATA = "CSV_Files/Silver Data Last Month.csv"
#make life easier by assigning more specific working variables,
#so easier to change which stocks to analyze

current_train_data = SILVER_TRAIN_DATA
current_test_data = SILVER_TEST_DATA

# train is from year, test is from month
NUM_TRAIN_DATA_POINTS = 309
NUM_TEST_DATA_POINTS = 27

LEARNING_RATE = 0.1
NUM_EPOCHS = 5000

# function to load and filter data from csv

def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name, # the stock we're using
                       skiprows = 0, #headings skipped
                       nrows=num_data_points,# the amount of info we have
                       usecols=['Price','Open','Vol.']) # the columns we're focusing on
    # prices of the stocks each day
    final_prices = data['Price'].astype(str).str.replace(',','').astype(np.float)# format without comma to prevent probs
    # price at beginning of the day
    opening_prices = data['Open'].astype(str).str.replace(',','').astype(np.float)
    # volumes exchanged throughout the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes


# fucntion to calc price differences
def calculate_price_differences(final_prices, opening_prices):
    price_differences = []
    for diff in range(len(opening_prices) - 1):
        price_difference = opening_prices[diff + 1] - final_prices[diff]
        price_differences.append(price_difference)
    return price_differences

def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100


# training data sets

train_final_prices, train_opening_prices, train_volumes = load_stock_data(current_train_data, NUM_TRAIN_DATA_POINTS)
train_price_differences = calculate_price_differences(train_final_prices, train_opening_prices)
train_volumes = train_volumes[:-1]

# testing data sets
test_final_prices, test_opening_prices, test_volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
test_price_differences = calculate_price_differences(test_final_prices, test_opening_prices)
test_volumes = test_volumes[:-1]

# y = Wx + b

x = tf.placeholder(tf.float32, name = 'x')
W = tf.Variable([0.1], name = 'W')
b = tf.Variable([0.1], name = 'b')
y = W * x + b

y_predicted = tf.placeholder(tf.float32, name = 'y_predicted')

loss = tf.reduce_sum(tf.square(y - y_predicted))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
for _ in range(NUM_EPOCHS):
    session.run(optimizer, feed_dict = {x: train_volumes, y_predicted: train_price_differences})

results = session.run(y, feed_dict={x: test_volumes})
accuracy = calculate_accuracy(test_price_differences, results)
print("Accuracy of model: {0:.2f}%".format(accuracy))

#for plotting
#plt.figure(1)
#plt.plot(train_volumes, train_price_differences, 'bo')
#plt.title("Price Differences For Given Volumes for the Past Year")
#plt.xlabel("Volumes")
#plt.ylabel("Price Differences")
#plt.show()
