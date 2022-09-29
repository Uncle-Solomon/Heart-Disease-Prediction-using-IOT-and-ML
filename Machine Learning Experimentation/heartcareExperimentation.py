## Execute each block of code in jupyter notebook
#Block1: Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb

%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

#Support Vector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# Block2
dataset = pd.read_csv('New_Dataset.csv')
dataset

# Block3
dataset['target'].value_counts().plot(kind='bar', color=['red', 'green'] );

# Block4
corr_matrix = dataset.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = snb.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='YlGnBu');

# Block5
# Splitting data into features and labels

x = dataset.drop('target', axis=1) 

y =dataset['target']

np.random.seed(42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Block6
model_dict = {'Logistic Regression': LogisticRegression(),
              'KNN': KNeighborsClassifier(),
              'Random Forest': RandomForestClassifier(),
              'Gradient Boost': GradientBoostingClassifier(),
              'AdaBoost': AdaBoostClassifier(),
              'Extra Tree': ExtraTreesClassifier(),
             }

def fit_and_score (models, x_train, x_test, y_train, y_test):
    
    np.random.seed(42)
    model_accuracy = {}
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_accuracy[name] = model.score(x_test, y_test)
        
    return model_accuracy

# Block7
model_accuracy = fit_and_score(model_dict, x_train, x_test, y_train, y_test)
model_accuracy

# Block 8
np.random.seed(48)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

# Block 9
model_accuracy['SVC'] = clf.score(x_test, y_test)
model_accuracy

# Block 10
model_compare = pd.DataFrame(model_accuracy, index=['Accuracy on Test Data'])

model_compare.T.plot(kind='bar', figsize=(10,6));
#plt.yticks()
plt.xticks(rotation=0)

# Block 11
model_accuracy2 = fit_and_score(model_dict, x_train, x_train, y_train, y_train)
model_accuracy2

# Block 12
model_accuracy2['SVC'] = clf.score(x_train, y_train)
model_accuracy2

# Block 13
model_compare = pd.DataFrame(model_accuracy2, index=['Accuracy on Training Data'])

model_compare.T.plot(kind='bar', figsize=(10,6));
#plt.yticks()
plt.xticks(rotation=0)

# Block 14
import warnings
warnings.filterwarnings('ignore')


clf_a = LogisticRegression(random_state=0)
clf_a.fit(x_train, y_train)
clf_a.score(x_test, y_test)

y_pred_a = clf_a.predict(x_test)
print(classification_report(y_test, y_pred_a))

clf_b = KNeighborsClassifier(n_neighbors=3)
clf_b.fit(x_train, y_train)
clf_b.score(x_test, y_test)

y_pred_b = clf_b.predict(x_test)
print(classification_report(y_test, y_pred_b))

np.random.seed(6)
clf_c = RandomForestClassifier(n_estimators=100)
clf_c.fit(x_train, y_train)

y_pred_c = clf_c.predict(x_test)
print(classification_report(y_test, y_pred_c))

clf_d = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf_d.fit(x_train, y_train)
clf_d.score(x_test, y_test)

y_pred_d = clf_d.predict(x_test)
print(classification_report(y_test, y_pred_d))

clf_e = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_e.fit(x_train, y_train)
clf_e.score(x_test, y_test), clf_e.score(x_train, y_train)

y_pred_e = clf_e.predict(x_test)
print(classification_report(y_test, y_pred_e))

plot_roc_curve(clf_e, x_test, y_test)
plt.show()

snb.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    
    fig, ax = plt.subplots(figsize= (5,5))
    ax = snb.heatmap(confusion_matrix(y_test, y_preds), annot=True,cbar=False)
    
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    
plot_conf_mat(y_test, y_pred_e)

clf_f = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_f.fit(x_train, y_train)
clf_f.score(x_test, y_test)

y_pred_f = clf_f.predict(x_test)
print(classification_report(y_test, y_pred_f))

y_pred_svc = clf.predict(x_test)
print(classification_report(y_test, y_pred_svc))

np.random.seed(42)
clf_g = tree.DecisionTreeClassifier()
clf_g.fit(x_train, y_train)
clf_g.score(x_test, y_test)

y_pred_g = clf_g.predict(x_test)
print(classification_report(y_test, y_pred_g))

clf_h = MLPClassifier(random_state=1, max_iter=300)
clf_h.fit(x_train, y_train)
clf_h.score(x_test, y_test)

y_pred_h = clf_h.predict(x_test)
print(classification_report(y_test, y_pred_h))

from sklearn.naive_bayes import GaussianNB
clf_i = GaussianNB()
clf_i.fit(x_train, y_train)
clf_i.score(x_test, y_test)

y_pred_i = clf_i.predict(x_test)
print(classification_report(y_test, y_pred_i))

compare_acc = {'KNN': clf_b.score(x_test, y_test),
              'Decision Tree': clf_g.score(x_test, y_test),
              'Naive-Bayes': clf_i.score(x_test, y_test),
              'Logistic Regression': clf_a.score(x_test, y_test),
              'Random Forest': clf_c.score(x_test, y_test),
              'Gradient Boost': clf_d.score(x_test, y_test),
              'Support Vector': clf.score(x_test, y_test),
              'Multi-Layer Perceptron': clf_h.score(x_test, y_test),
              'AdaBoost': clf_e.score(x_test, y_test)
             }
model_compare = pd.DataFrame(compare_acc, index=['Accuracy on Test Data'])

model_compare.T.plot(kind='bar', figsize=(20,10));
#plt.yticks()
plt.xticks(rotation=0)


import joblib as jb

jb.dump(clf, 'model.pkl')


import pickle
with open('model_two.pkl', 'wb') as file:
    pickle.dump(clf, file)

