import pandas as pd
import numpy as np
df = pd.read_csv("haberman.csv")

def prediction(feature_train,feature_test,label_train,label_test):

    print("feature_train",feature_train.shape)
    print("label_train",label_train.shape)
    print("feature_test",feature_test.shape)
    print("label_test",label_test.shape)
    print("feature train:\n",feature_train)
    print("label train:\n",label_train)
    from sklearn import svm

    clf = svm.SVC(kernel='linear', C=1).fit(feature_train, label_train)
    svm_scr=clf.score(feature_test, label_test)
    print("the SVM score is : ",svm_scr)


    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(feature_train,label_train)
    gnb_scr=gnb.score(feature_test, label_test)
    print("the Guassian NB score is : ",gnb_scr)

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(feature_train,label_train)
    tree_scr=tree.score(feature_test, label_test)
    print("decision tree score is :",tree_scr)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    rf.fit(feature_train,label_train)
    rf_scr=rf.score(feature_test, label_test)
    print("the random Forest score is : ",rf_scr)


    from sklearn.ensemble import AdaBoostClassifier
    ab = AdaBoostClassifier(n_estimators=100, random_state=0)
    ab.fit(feature_train,label_train)
    ab_scr=ab.score(feature_test, label_test)
    print("the adaboost score is : ",ab_scr)


def dataPreProcess(df,label_name):
    label=df[label_name]
    feature=df.drop(label_name, axis=1)
    #splitting the data
    from sklearn.model_selection import train_test_split
    label_train, label_test, feature_train, feature_test = train_test_split(label,feature, test_size=0.4, random_state=0)
    #converting string types into float type
    
    feature_train=feature_train.astype(np.float)
    feature_test=feature_test.astype(np.float)
    return label_train, label_test, feature_train, feature_test


label_train, label_test, feature_train, feature_test= dataPreProcess(df,"status")
prediction(feature_train,feature_test,label_train,label_test)
