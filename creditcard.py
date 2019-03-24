import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
import sklearn

data=pd.read_csv('creditcard.csv')
print(data.columns)
print(data.shape)
print(data.describe())
data=data.sample(frac=0.2,random_state=1)
print(data.shape)
data.hist(figsize=(20,20))
plt.show()

fraud=data[data['Class']==1]
valid=data[data['Class']==0]
outlier_frac=len(fraud)/float(len(valid))
print(outlier_frac)
print("fraud:{}".format(len(fraud)))
print("valid:{}".format(len(valid)))

#correlation matrix
corr_matrix=data.corr()
fig = plt.figure()
sns.heatmap(corr_matrix,square=True)
plt.show()


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
X=data[columns]
Y=data[target]

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state 
state=1
#define outlier detection methods
classifiers={"Isolation Forest": IsolationForest(max_samples=len(X),
                                                contamination=outlier_frac,random_state=state),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,contamination=outlier_frac)}

#fit the model
n_outliers=len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name=="Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        score_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        score_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
    #reshape predictor values 1 for fraud and 0 for valid
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    no_errors=(y_pred!=Y).sum()
    
    #run classification metrics
    print("{}:{}".format(clf_name,no_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
