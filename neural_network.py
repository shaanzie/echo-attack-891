#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:21:30 2023

@author: kleveri2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:10:31 2023

@author: kleveri2
"""

# !pip install --upgrade scipy pandas
import scipy.io
import pandas as pd
final_df = pd.DataFrame()
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler



directory_path = '../siri'
y2 = []
x2 = []
# Walk through all files and subdirectories in the specified directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # Full path of the file
        file_path = os.path.join(root, file_name)

        if file_path.endswith('.mat') and 'idle' not in file_path and 'MAC' not in file_path:
            try:
                mat_data = scipy.io.loadmat(file_path, mat_dtype=True)
                cols = [i for i in range(1, 1601)]
                df = pd.DataFrame(mat_data['data'], columns = cols)
                df['label'] = 0

                final_df = pd.concat([final_df, df], ignore_index=True)
            except:
                print('File: ' + file_path + ' not processed!')
import os
'''
directory_path = '../pixel'

# Walk through all files and subdirectories in the specified directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # Full path of the file
        file_path = os.path.join(root, file_name)

        if file_path.endswith('.mat'):
            try:
                mat_data = scipy.io.loadmat(file_path, mat_dtype=True)
                cols = [i for i in range(1, 1601)]
                df = pd.DataFrame(mat_data['data'], columns = cols)
                df['label'] = 0
                final_df = pd.concat([final_df, df], ignore_index=True)
                print('File: ' + file_path + '  processed!')

            except:
                print('File: ' + file_path + ' not processed!')
                
directory_path = '../bixby'

# Walk through all files and subdirectories in the specified directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # Full path of the file
        file_path = os.path.join(root, file_name)

        if file_path.endswith('.mat'):
            try:
                mat_data = scipy.io.loadmat(file_path, mat_dtype=True)
                cols = [i for i in range(1, 1601)]
                df = pd.DataFrame(mat_data['data'], columns = cols)
                df['label'] = 0
                final_df = pd.concat([final_df, df], ignore_index=True)
                print('File: ' + file_path + '  processed!')

            except:
                print('File: ' + file_path + ' not processed!')

directory_path = '../alexa'

# Walk through all files and subdirectories in the specified directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # Full path of the file
        file_path = os.path.join(root, file_name)

        if file_path.endswith('.mat'):
            try:
                mat_data = scipy.io.loadmat(file_path, mat_dtype=True)
                cols = [i for i in range(1, 1601)]
                df = pd.DataFrame(mat_data['data'], columns = cols)
                df['label'] = 0
                final_df = pd.concat([final_df, df], ignore_index=True)
                print('File: ' + file_path + '  processed!')

            except:
                print('File: ' + file_path + ' not processed!')
'''
directory_path = '../idle'

# Walk through all files and subdirectories in the specified directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # Full path of the file
        file_path = os.path.join(root, file_name)

        if file_path.endswith('.mat'):
            try:
                mat_data = scipy.io.loadmat(file_path, mat_dtype=True)
                cols = [i for i in range(1, 1601)]
                df = pd.DataFrame(mat_data['data'], columns = cols)
                df['label'] = 1
                final_df = pd.concat([final_df, df], ignore_index=True)
                print('File: ' + file_path + '  processed!')

            except:
                print('File: ' + file_path + ' not processed!')



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier




target_column = 'label'
total = 0

X = final_df.drop(target_column, axis=1)
y = final_df[target_column]


x2 = []
y2 = []
temp = []
temp2 = []
for index, row in X.iterrows():
    for j in row:
        temp.append(j)
    #print(np.array(row))
    #for i in range(1599):
    #    temp.append(row[i+1])
    #temp2.append(np.array(row))
    #temp.clear()
    if index % 16 == 0:
        if len(temp) < 25600:
            m = 0
            temp.clear()
            continue
        x2.append(temp[:])
        temp.clear()

        temp2.clear()
        y2.append(y[index])
        
#print(x2)
'''
X = df.values.tolist()
xcount = 0
temp = []

x2 = []
y2 = []
for i in range(len(X)):
    total = total + 1
    temp.append(X[i])
    if len(temp) == 16:
        x2.append(temp[:])
        y2.append(y[total])
        temp = []
        '''
print(len(x2))
print(len(y2))


#print(X)

print(x2[1])
print(len(x2[-1]))
#x3 = np.array(x2)


print(y2)
X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print('Original dataset shape %s' % Counter(y_train))

# .76
#d_tree = DecisionTreeClassifier(criterion="gini", max_features="auto")
#d_tree = MLPClassifier(random_state=1, batch_size=8, max_iter=300, activation="relu", solver="adam",learning_rate_init=.001)
d_tree = MLPClassifier(random_state=1, max_iter=300, activation="relu", solver="adam")

#d_tree = DecisionTreeClassifier()

d_tree.fit(X_train_resampled, y_train_resampled)



y_pred = d_tree.predict(X_test)
print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
Accuracy: 0.76
tree_rules = export_text(d_tree, feature_names=list(X.columns))
#print(tree_rules)