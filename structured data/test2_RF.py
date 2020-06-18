"""
We trained a Random Forest classifier (80 trees, max depth is 5, minimum 5
samples per leaf) on original dataset. Then we used divideup framework to
partition the dataset based on some rationales and train the RF classifier.
"""
# projection function
def projection(model,Xtest,Ytest):
    tt=model.predict(Xtest)
    count = 0;
    for ii in range(0,tt.shape[0]):
        if(Ytest[ii]==0 or Ytest[ii]==3):
            if(tt[ii]==0 or tt[ii]==3): count=count+1
        if(Ytest[ii]==1 and tt[ii]==1):
            count=count+1
    return count/tt.shape[0]

# read data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
data=np.loadtxt('transfusion.data',str,delimiter=',')
data=data.astype('int32')
data_y=data[:,data.shape[1]-1]
data_x=data[:,0:data.shape[1]-1]
np.random.seed(666)

# original RF classifier
rfc = RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=5)
from sklearn.model_selection import RepeatedKFold
kf = RepeatedKFold(n_splits=10, n_repeats=10)
sum=0
for train_index, test_index in kf.split(data_x,data_y):
    Xtrain,Ytrain=data_x[train_index],data_y[train_index]
    Xtest,Ytest=data_x[test_index],data_y[test_index]
    rfc=rfc.fit(Xtrain,Ytrain)
    score_r = rfc.score(Xtest, Ytest)
    sum+=score_r
print("Random Forest:{}".format(sum/100))

# partition the dataset
for ii in range(0, data_x.shape[0]):
    if (data_x[ii][0] <=6 and data_y[ii] == 0):
        data_y[ii] =3

# learning process and prediction accuracy of RF
sum=0
rfc_d = RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=5)
for train_index, test_index in kf.split(data_x,data_y):
    Xtrain,Ytrain=data_x[train_index],data_y[train_index]
    Xtest,Ytest=data_x[test_index],data_y[test_index]
    rfc_d=rfc_d.fit(Xtrain,Ytrain)
    score_r = projection(rfc_d,Xtest,Ytest)
    sum+=score_r
print("Random Forest:{} divideup".format(sum/100))



