#!/usr/bin/python

import sys
import pickle
import pandas as pd
import matplotlib.pyplot

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


##FUNCTIONS
#Ploting our data in 2D plot
#input dataset, feature to plot on X and feature to plot on Y
def plotFeatures(data, xcoord, ycoord):
    for point in data:
        salary = point[xcoord]
        bonus = point[ycoord]
        matplotlib.pyplot.scatter( salary, bonus )

    matplotlib.pyplot.xlabel(features_list[xcoord])
    matplotlib.pyplot.ylabel(features_list[ycoord])
    matplotlib.pyplot.show()


#Fitting given classifier on given data
#printing score, recall and precission
#returning fitted classifier
def Classify(clas, features_train, features_test, labels_train, labels_test):
    clf = clas
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print("These are results from {} using all selected features".format(clas))
    print("Score: {}".format(clf.score(features_test, labels_test)))
    print("Precision: {}".format(precision_score(labels_test, pred)))
    print("Recall: {}".format(recall_score(labels_test, pred)))

    print(labels_test)
    print("predictions:")
    print(pred)

    #how many is true positive and false negative
    truep = 0
    falsen = 0
    for predicted,label in zip(pred,labels_test):
        if (predicted == 1.0) & (label == 1.0): truep +=1
        elif (predicted == 0.0) & (label == 1.0): falsen +=1
    print("No of true positive: {}".format(truep))
    print("No of missed positive (=false negative): {}".format(falsen))

    return clf



##END OF FUNCTIONS


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus',  'bonus_to_salary', 'from_poi_weighted', 'to_poi_weighted', 'shared_poi_weighted'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Putting data into dataframe and do some basic diagnostics
dataf = pd.DataFrame.from_dict(data_dict, orient='index')
print(dataf.columns)

print("Size of original sample:")
print(dataf.shape)

print("Number of POIs:")
print(dataf[dataf["poi"]==1].shape)


#create floats from columns I am going to use in analysis
columns = ["salary", "bonus", "from_poi_to_this_person", "to_messages", "from_this_person_to_poi", "from_messages", "shared_receipt_with_poi"]
for c in columns:
    dataf[c] = dataf[c].apply(pd.to_numeric, errors='coerce')

#creating new feautres
#ratio of bonuses to salary. maybe POI are getting improportionally high bonuses
dataf["bonus_to_salary"] = dataf["bonus"]/dataf["salary"]

#emails from/to/shared with POI should be weighted by total amount of emails of that type
dataf["from_poi_weighted"] = dataf["from_poi_to_this_person"]/dataf["to_messages"]
dataf["to_poi_weighted"] = dataf["from_this_person_to_poi"]/dataf["from_messages"]
dataf["shared_poi_weighted"] = dataf["shared_receipt_with_poi"]/dataf["to_messages"]


#check some descriptive statistics of the features I want to use
for c in columns:
    print("Feature: {} has minimum: {}, maximum: {}, and mean: {}".format(c, dataf[c].min(),
        dataf[c].max(), dataf[c].mean()))

#there are too extreme maximas for salaries and bonuses, check it
print("extreme salary value")
print(dataf['salary'].nlargest(5).index.tolist())

print("extreme bonus value")
print(dataf['bonus'].nlargest(5).index.tolist())

#we see it is entry called TOTAL, that was there left from the figure, let's get rid of it
dataf = dataf.drop(["TOTAL"])


#Biggest outliers in terms of salary and bonuses, but dropping them doesn't help anything.
#There is also no intuition why they should be dropped. They are not caused by errors in data
#dataf = dataf.drop(["SKILLING JEFFREY K"])
#dataf = dataf.drop(["LAY KENNETH L"])
#dataf = dataf.drop(["LAVORATO JOHN J"])



#here we drop entries with missing values in the columns we are interested in for later analysis
#keeping full dataset as well
fulldataf = dataf.copy()
dataf.dropna(subset = features_list, inplace=True)

print("Number of datapoints remained after droping NaN:")
print(dataf.shape)

print("out of it POIs:")
print(dataf[dataf["poi"]==1].shape)

#check if the cleaned dataset contains only names of people or if there is any suspicious ID
print(dataf.index.tolist())

#looks OK

#later we can try with full dataset, where mean values replaced all NaN
fulldataf = fulldataf.fillna(fulldataf.mean())


#putting everything back to dictionary
data_dict = dataf.to_dict(orient='index')
fulldata_dict = fulldataf.to_dict(orient='index')
### Task 2: Remove outliers
#so we removed already observation called TOTAL, let's check how the data looks in some plots

data = featureFormat(data_dict, features_list)
fulldata = featureFormat(fulldata_dict, features_list)

#plotting salary x bonuses
#and both features separately with POI, we have extreme values for both POI and nonPOI
#plotFeatures(data,1,2)
#plotFeatures(data,0,1)
#plotFeatures(data,0,2)


#we still have a handful of outliers, but for now I would keep them in the dataset
#later we can try to rerun our model without them to see how it influences our results


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#preparing also full dataset, replacing NaN values with median values
fulldata = featureFormat(fulldata_dict, features_list, sort_keys = True)
fulllabels, fullfeatures = targetFeatureSplit(fulldata)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Let's try few clasifiers. It is all about supervised classifiers


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, stratify=labels)



clf = Classify(GaussianNB(), features_train, features_test, labels_train, labels_test)
#not suprisingly poor results

#let's try decision tree and also random forest
clf = Classify(DecisionTreeClassifier(), features_train, features_test, labels_train, labels_test)

#Random forrest
clf = Classify(RandomForestClassifier(n_estimators=1000, criterion="entropy", min_samples_split=2), features_train, features_test, labels_train, labels_test)


#Let's try random forest with grid search
print "Random forrest with grid search"
param_grid = {
          'min_samples_split': [2, 5, 10],
          'max_features' : [1,2,"auto", None],
          }

clf = Classify(GridSearchCV(RandomForestClassifier(criterion="entropy", n_estimators=1000), param_grid), features_train, features_test, labels_train, labels_test)

#save only best estimator for evaluation of solution
clf = clf.best_estimator_

print("The best estimator chosen in grid search:")
print(clf)

print("Importance of features:")
print(clf.feature_importances_)


#A little bit of crossvalidation, to see if our result is not just driven by specific train and test subsamples

n_samples = len(features)
cv = ShuffleSplit(n_splits=30, test_size=0.3, random_state=0)
precision = cross_val_score(clf, features, labels, cv=cv, scoring="precision_macro")
recall = cross_val_score(clf, features, labels, cv=cv, scoring="recall_macro")
print("Precission on different subsets")
print("{} with average {}".format(precision, precision.mean()))
print("Recall on different subsets")
print("{} with average {}".format(recall, recall.mean()))




#let's see how it works with full dataset where we used mean values to replace all NaN
#with imputed means, the classifier has much lower precision, without improving recall

#features_train, features_test, labels_train, labels_test = \
#    train_test_split(fullfeatures, fulllabels, test_size=0.3, random_state=42)

#clf = Classify(DecisionTreeClassifier(), features_train, features_test, labels_train, labels_test)







### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
