import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('laptop_scrap_data.csv')

df['Inches'] = df['Inches'].fillna(df['Inches'].mode()[0])
df['Price'] = df['Price'].fillna(df['Price'].mode()[0])
df['Ram_GB'] = df['Ram_GB'].fillna(df['Ram_GB'].mode()[0])
df['SSD'] = df['SSD'].fillna(df['SSD'].mode()[0])
df['Company'] = df['Company'].fillna(df['Company'].mode()[0])
df['OpSys'] = df['OpSys'].fillna(df['OpSys'].mode()[0])
df = df.dropna()

df.drop('TypeName', axis=1, inplace=True)
df.drop('ScreenResolution', axis=1, inplace=True)
df.drop('Cpu', axis=1, inplace=True)
df.drop('Gpu', axis=1, inplace=True)

x = df.iloc[:, 0:-2]
print(x.shape)
y = df['Storage_Category']
print(y.shape)

cat_columns = x.select_dtypes(['str']).columns
x[cat_columns] = x[cat_columns].apply(lambda col: pd.factorize(col)[0])
y = pd.factorize(y)[0]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.3, shuffle=False)
print('Train:', X_train.shape)
print('Test:', X_test.shape)

#RandomForestModel
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
Y_rpred = RF.predict(X_test)
r_accuracy = metrics.accuracy_score(Y_test, Y_rpred)
print('Random Forest Accuracy: %f' % r_accuracy)
r_precision = metrics.precision_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('Precision: %f' % r_precision)
r_recall = metrics.recall_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('Recall: %f' % r_recall)
r_f1 = metrics.f1_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print('F1 score: %f' % r_f1)

#Decision Tree
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, Y_train)
Y_dpred = Dtree.predict(X_test)
d_accuracy = metrics.accuracy_score(Y_test, Y_dpred)
print('Decision Tree Accuracy: %f' % d_accuracy)
d_precision = metrics.precision_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('Precision: %f' % d_precision)
d_recall = metrics.recall_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('Recall: %f' % d_recall)
d_f1 = metrics.f1_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print('F1 score: %f' % d_f1)

#GaussianNB
GausNB = GaussianNB()
GausNB.fit(X_train, Y_train)
Y_gpred = GausNB.predict(X_test)
g_accuracy = metrics.accuracy_score(Y_test, Y_gpred)
print('Gaussian Accuracy: %f' % g_accuracy)
g_precision = metrics.precision_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
print('Precision: %f' % g_precision)
g_recall = metrics.recall_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
print('Recall: %f' % g_recall)
g_f1 = metrics.f1_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
print('F1 score: %f' % g_f1)


#accuracy graph
left = [1, 2, 3]
height = [r_accuracy, d_accuracy, g_accuracy,]
tick_label = ['Random Forest', 'Decision Tree', 'Gaussian']
plt.figure(figsize=(10, 6))
plt.bar(left, height, tick_label=tick_label, width=0.7, color=['#08737f', '#089f8f', '#64c987'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy of Applied Classifiers')
plt.show()