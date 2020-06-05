# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

df = pd.read_csv(r'C:\Users\admin\Desktop\files\company.csv')
x = df.iloc[:, [2,3]].values
y = df.iloc[:, 4].values

# Splitting the dataset into training and test se

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

#feature Scaling  

from sklearn.preprocessing import StandardScaler    
s_c= StandardScaler()    
x_train= s_c.fit_transform(x_train)    
x_test= s_c.transform(x_test)     

#Support vector classifier 

from sklearn.svm import SVC  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)  

#Predicting the test set result  

y_pred= classifier.predict(x_test)  

#Creating the Confusion matrix 

from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred) 

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red', 'green')))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
plt.title('SVM classifier (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  

#Visulaizing the test set result 

from matplotlib.colors import ListedColormap  
x_set, y_set = x_test, y_test  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
plt.title('SVM classifier (Test set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  





