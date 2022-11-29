import csv

import numpy as np


def take_data():
    with open('winequality-red.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        y = [row[11] for row in reader]
        y.pop(0)
    with open('winequality-red.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        x = [[row[i] for i in range(11)] for row in reader]
        x.pop(0) 
	
        x_test = np.array(x)
        y_test = np.array(y)
    
    with open('winequality-white.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        y = [row[11] for row in reader]
        y.pop(0)
    with open('winequality-white.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        x = [[row[i] for i in range(11)] for row in reader]
        x.pop(0)
	
        x_train = np.array(x)
        y_train = np.array(y)  
    
    return (x_train, y_train), (x_test, y_test)
