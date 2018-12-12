import numpy as np
from matplotlib import pyplot
from sklearn import model_selection, svm, linear_model, ensemble, metrics
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import itertools
import scikitplot as skplt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import KFold
import seaborn as sns


dataset = read_csv('oasis_longitudinal.csv', skiprows=[0],header=None)
dataset[[9,10]] = dataset[[9,10]].replace("", np.NaN)
#print(dataset)
values = dataset[[9,10]]

imputer=SimpleImputer(missing_values=np.nan, strategy='median')
transformed_values = imputer.fit_transform(values)
#count the number of NaN values in each column
#print(np.isnan(transformed_values).sum())

dataset[[9,10]]=transformed_values
dataset[15]=0
dataset[15] = dataset[11].apply(lambda x: 1 if x>=0.5 else 0)



#Visualizing the data
data=dataset.rename(columns={5: 'Gender', 7:'Age', 11: 'CDR'})
data=data[["Gender","Age","CDR"]]
sns.violinplot(x="Gender",y="CDR", data=data, palette="muted")
plt.show()

CDR = dataset[11]

Age = dataset[7]
plt.figure(1, figsize=(9, 4))
plt.subplot(132)
plt.scatter(Age, CDR)
plt.xlabel('Age')
plt.ylabel('CDR')
plt.show()

Educ =dataset[8]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(Educ, CDR)
plt.xlabel('Education')
plt.ylabel('CDR')
plt.show()

SES =dataset[9]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(SES, CDR)
plt.xlabel('SES')
plt.ylabel('CDR')
plt.show()

MMSE =dataset[10]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(MMSE, CDR)
plt.xlabel('MMSE')
plt.ylabel('CDR')
plt.show()

eTIV =dataset[12]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(eTIV, CDR)
plt.xlabel('eTIV')
plt.ylabel('CDR')
plt.show()

nWBV =dataset[13]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(nWBV, CDR)
plt.xlabel('nWBV')
plt.ylabel('CDR')
plt.show()

ASF =dataset[14]
plt.figure(1, figsize=(9, 5))
plt.subplot(132)
plt.scatter(ASF, CDR)
plt.xlabel('ASF')
plt.ylabel('CDR')
plt.show()

dataset[5] = dataset[5].astype("category").cat.codes # convert M&F to numerical
dataset=dataset.drop([0,1,2,3,4,6,11], axis=1)
#print(dataset)

set=dataset.values

x = dataset.loc[:,0:14]
y = dataset.loc[:,15]
valid_size = 0.3
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x,y, test_size=valid_size)


#Training diffrent ML classifiers
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("Accuracy of KNN: ",accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
scores = cross_val_score(knn, x, y, cv=5)
print("THE SCORE IS: ", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Logistic Regression with l2 Regularization
LR =LogisticRegression(penalty='l1')
LR.fit(X_train,Y_train)
predictions2=LR.predict(X_validation)
print("Accuracy of Logistic Regression: ",accuracy_score(Y_validation, predictions2))
print(confusion_matrix(Y_validation, predictions2))
print(classification_report(Y_validation, predictions2))



clf= ensemble.GradientBoostingClassifier(n_estimators= 1000, max_leaf_nodes= 4, random_state= 2, learning_rate=0.1, min_samples_split= 5,)
clf.fit(X_train, Y_train)
predictions3=clf.predict(X_validation)
print("Accuracy of Gradient Boosting: ",accuracy_score(Y_validation, predictions3))
print(confusion_matrix(Y_validation, predictions3))
print(classification_report(Y_validation, predictions3))

clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf2.fit(X_train,Y_train)
predictions4=clf2.predict(X_validation)
print("Accuracy of Decision Tree: ",accuracy_score(Y_validation, predictions4))
print(classification_report(Y_validation, predictions4))
print("Feature Importance: ", clf2.feature_importances_)

bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train,Y_train)
predictions5=bagging.predict(X_validation)
print("Accuracy of Bagging Classifier: ",accuracy_score(Y_validation, predictions5))
print(classification_report(Y_validation, predictions5))

adaboost= AdaBoostClassifier(n_estimators=100)
#score = cross_val_score(adaboost, X_train, Y_train, cv=10)
adaboost.fit(X_train,Y_train)
predictions6=adaboost.predict(X_validation)
print("Accuracy of Adaboost: ",accuracy_score(Y_validation, predictions6))
print(classification_report(Y_validation, predictions6))

SVM = svm.SVC(gamma='scale',probability =True, C=8)
SVM.fit(X_train, Y_train)
predictions7=SVM.predict(X_validation)
print("Accuracy of SvM: ",accuracy_score(Y_validation, predictions7))
print(classification_report(Y_validation, predictions7))
#print(score.mean())

def plot_confusion_matrix(cm, classes, normalize, title='Confusion matrix',cmap=plt.cm.Blues): # retreived from Sklearn documentation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
conf =confusion_matrix(Y_validation, predictions7)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of SVM')
plt.show()

conf =confusion_matrix(Y_validation, predictions)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of KNN')
plt.show()

conf =confusion_matrix(Y_validation, predictions2)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of Logistic Regression')
plt.show()

conf =confusion_matrix(Y_validation, predictions3)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of Gradient Boosting')
plt.show()

conf =confusion_matrix(Y_validation, predictions4)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of Decision Trees')
plt.show()

conf =confusion_matrix(Y_validation, predictions6)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf, classes=["Demented","Non-demented"], normalize =True, title='Confusion matrix of Adaboost')
plt.show()


def plotROC(classifier, X_validation, title): # retreived from Sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
  probs = classifier.predict_proba(X_validation)
# keep probabilities for the positive outcome only
  probs = probs[:, 1]
  auc = roc_auc_score(Y_validation, probs)
  print('AUC: %.3f' % auc)
# calculate roc curve
  fpr, tpr, thresholds = metrics.roc_curve(Y_validation, probs)
# plot no skill
  pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
  pyplot.plot(fpr, tpr, marker='.')
  pyplot.title(title)
# show the plot
  pyplot.show()
  #print(Y_validation)
  #print(probs)


plotROC(LR, X_validation, "Logistic Regression")
plotROC(knn, X_validation, "KNN")
plotROC(SVM, X_validation, "SVM")

#K-fold cross-validation
kf = KFold(n_splits=10)
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator

print(kf)
#x=np.array(x.to_records(index="False"))
#y=np.array(y.values)

print (type(x))
print (type(y))
#print (np.shape(x))
#print (np.shape(y))
for train_index, test_index in kf.split(x):
# print("TRAIN:", train_index, "TEST:", test_index)
 x_train, x_test = x.loc[train_index], x.loc[test_index]
 y_train, y_test = y.loc[train_index], y.loc[test_index]
 clf = ensemble.GradientBoostingClassifier(n_estimators=1000, max_leaf_nodes=4, random_state=2, learning_rate=0.1,
                                          min_samples_split=5, )
 clf.fit(x_train, y_train)
 predictions3 = clf.predict(x_test)
 print("Accuracy of Gradient Boosting: ", accuracy_score(y_test, predictions3))
 print(confusion_matrix(y_test, predictions3))
 print(classification_report(y_test, predictions3))


scores = cross_val_score(clf, x, y, cv=9)
print ("Cross-validated scores:", scores)
print ("Mean score: ", np.mean(scores))