from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from copy import deepcopy
        
        
class BaseTClassifier(object):
    def __init__(self,learner0,learner1,type_=0):
        self.learner0 = learner0
        self.learner1 = learner1
        self.model0 = None
        self.model1 = None
        self.type = type_

    def fit(self,X,y,treatment):
        X0 = X[treatment==0]
        X1 = X[treatment==1]
        y0 = y[treatment==0]
        y1 = y[treatment==1]

        self.model0 = self.learner0.fit(X0, y0)
        self.model1 = self.learner1.fit(X1, y1)

    def predict(self,X, idx):
        if idx==0:
            if self.type==0:
                prob = self.model0.predict_proba(X)[:, 1]
            else:
                prob = self.model0.predict(X).reshape(-1)
        else:
            if self.type==0:
                prob = self.model1.predict_proba(X)[:, 1]
            else:
                prob = self.model1.predict(X).reshape(-1)
        return prob

    def get_cate(self,X,y,treatment):
        # replace all with estimated value
        self.fit(X,y,treatment)
        prob0 = self.predict(X, idx=0)
        prob1 = self.predict(X, idx=1)
        return prob1-prob0

    def get_cate_(self,X,y,treatment):
        # not replace the existing value
        self.fit(X,y,treatment)
        prob0 = self.predict(X[treatment==1], idx=0)
        prob1 = self.predict(X[treatment==0], idx=1)

        yhat_0 = np.zeros(len(y))
        yhat_1 = np.zeros(len(y))
        yhat_0[treatment==0]=y[treatment==0]
        yhat_0[treatment==1]=prob0
        yhat_1[treatment==1]=y[treatment==1]
        yhat_1[treatment==0]=prob1

        return yhat_1-yhat_0
    
    def get_cate2(self,Xtrain,ytrain,Xtest,len0, len1,flag=1):
        # replace all with estimated value
        if flag:
            self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
        prob0 = self.predict(Xtest, idx=0)
        prob1 = self.predict(Xtest, idx=1)
        return prob1-prob0
    
    def get_cate_2(self,Xtrain,ytrain,Xtest,ytest,len0, len1,treatment,flag=1):
        # not replace the existing value
        if flag:
            self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
        prob0 = self.predict(Xtest[treatment==1], idx=0)
        prob1 = self.predict(Xtest[treatment==0], idx=1)

        yhat_0 = np.zeros(len(ytest))
        yhat_1 = np.zeros(len(ytest))
        yhat_0[treatment==0]=ytest[treatment==0]
        yhat_0[treatment==1]=prob0
        yhat_1[treatment==1]=ytest[treatment==1]
        yhat_1[treatment==0]=prob1
    
        return yhat_1-yhat_0
    

class BaseSClassifier(object):
    def __init__(self,learner, type_=0):
        self.learner = learner
        self.model = None
        self.type = type_

    def fit(self,X,y,treatment):
        new_X = np.concatenate((X,treatment.reshape(-1,1)),axis=1)
        self.model = self.learner.fit(new_X, y)

    def predict(self,X):
        if self.type==0:
            prob = self.model.predict_proba(X)[:, 1]
        else:
            prob = self.model.predict(X).reshape(-1)
        return prob

    def get_cate(self,X,y,treatment):
        # replace all with estimated value
        self.fit(X,y,treatment)
        X0 = np.concatenate((X,np.zeros([X.shape[0],1])),axis=1)
        X1 = np.concatenate((X,np.ones([X.shape[0],1])),axis=1)
        prob0 = self.predict(X0)
        prob1 = self.predict(X1)
        return prob1-prob0

    def get_cate_(self,X,y,treatment):
        # not replace the existing value
        self.fit(X,y,treatment)
        X0 = np.concatenate((X,np.zeros([X.shape[0],1])),axis=1)
        X1 = np.concatenate((X,np.ones([X.shape[0],1])),axis=1)
        prob0 = self.predict(X0)
        prob1 = self.predict(X1)

        yhat_0 = np.zeros(len(y))
        yhat_1 = np.zeros(len(y))
        yhat_0[treatment==0]=y[treatment==0]
        yhat_0[treatment==1]=prob0[treatment==1]
        yhat_1[treatment==1]=y[treatment==1]
        yhat_1[treatment==0]=prob1[treatment==0]

        return yhat_1-yhat_0
    
    def get_cate2(self,Xtrain,ytrain,Xtest,len0, len1,flag=1):
        # replace all with estimated value
        if flag:
            self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
        X0 = np.concatenate((Xtest,np.zeros([Xtest.shape[0],1])),axis=1)
        X1 = np.concatenate((Xtest,np.ones([Xtest.shape[0],1])),axis=1)
        prob0 = self.predict(X0)
        prob1 = self.predict(X1)
        return prob1-prob0
    
    def get_cate_2(self,Xtrain,ytrain,Xtest,ytest,len0, len1,treatment,flag=1):
        # not replace the existing value
        n = len(Xtrain)//2
        if flag:
            self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
            
        X0 = np.concatenate((Xtest,np.zeros([Xtest.shape[0],1])),axis=1)
        X1 = np.concatenate((Xtest,np.ones([Xtest.shape[0],1])),axis=1)
        prob0 = self.predict(X0)
        prob1 = self.predict(X1)

        yhat_0 = np.zeros(len(ytest))
        yhat_1 = np.zeros(len(ytest))
        yhat_0[treatment==0]=ytest[treatment==0]
        yhat_0[treatment==1]=prob0[treatment==1]
        yhat_1[treatment==1]=ytest[treatment==1]
        yhat_1[treatment==0]=prob1[treatment==0]

        return yhat_1-yhat_0
    

class BaseXClassifier(object):
    def __init__(self,learner, cate_learner, prospensity_learner=RandomForestClassifier(), type_=0):
        self.learner0 = deepcopy(learner)
        self.learner1 = deepcopy(learner)
        self.learner2 = deepcopy(cate_learner)
        self.learner3 = deepcopy(cate_learner)
        self.prospensity_learner = deepcopy(prospensity_learner)
        self.model0 = None
        self.model1 = None
        self.type = type_
        
    def fit(self,X,y,treatment):
        X0 = X[treatment==0]
        X1 = X[treatment==1]
        y0 = y[treatment==0]
        y1 = y[treatment==1]
        if self.type==0:
            D1 = y1 - self.learner0.fit(X0, y0).predict_proba(X1)[:,1]
            D0 = self.learner1.fit(X1, y1).predict_proba(X0)[:,1] -y0
        else:
            D1 = y1 - self.learner0.fit(X0, y0).predict(X1).reshape(-1)
            D0 = self.learner1.fit(X1, y1).predict(X0).reshape(-1) -y0
        
        self.model1 = self.learner2.fit(X1,D1)
        self.model0 = self.learner3.fit(X0,D0)
        
        return D0,D1
        
    def predict(self,X):
        prob0 = self.model0.predict(X).reshape(-1)
        prob1 = self.model1.predict(X).reshape(-1)
        return prob0,prob1
    
    def get_prospensity(self, X, treatment):
        #if self.type==0:
        prospensity = self.prospensity_learner.fit(X, treatment).predict_proba(X)[:,1]
        #else:
        #    prospensity = self.prospensity_learner.fit(X, treatment).predict(X)
        return prospensity
    
    def get_cate(self,X,y,treatment,p=None):
        # replace all with estimated value
        D0,D1 = self.fit(X,y,treatment)
        prob0, prob1 = self.predict(X)
        if p==None:
            prospensity = self.get_prospensity(X,treatment)
        else:
            prospensity = p
        #print(prob0.shape,prob1.shape)
        return prospensity*prob0+(1-prospensity)*prob1
    
    def get_cate2(self,Xtrain,ytrain,Xtest,len0,len1, treatment, p=None):
        # replace all with estimated value
        D0,D1 = self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
        prob0, prob1 = self.predict(Xtest)
        if p==None:
            prospensity = self.get_prospensity(Xtest,treatment)
        else:
            prospensity = p
        return prospensity*prob0+(1-prospensity)*prob1
    
    def get_cate_(self,X,y,treatment, p=None):
        # not replace the existing value
        D0,D1 = self.fit(X,y,treatment)
        prob0, prob1 = self.predict(X)

        if p==None:
            prospensity = self.get_prospensity(X,treatment)
        else:
            prospensity = p
        
        Dhat_0 = []
        Dhat_1 = []
        cnt0 = cnt1= 0
        for i in range(len(y)):
            if treatment[i]==0:
                Dhat_0.append(D0[cnt0])
                cnt0+=1
                Dhat_1.append(prob1[i])
            if treatment[i]==1:
                Dhat_1.append(D1[cnt1])
                cnt1+=1
                Dhat_0.append(prob0[i])
  
        return prospensity*np.array(Dhat_0)+(1-prospensity)*np.array(Dhat_1) 
    def get_cate_2(self,Xtrain,ytrain,Xtest,ytest,len0,len1, treatment, p=None):
        # not replace the existing value
        D0,D1 = self.fit(Xtrain,ytrain,np.array([0]*len0+[1]*len1))
        prob0, prob1 = self.predict(Xtest)

        if p==None:
            prospensity = self.get_prospensity(Xtest,treatment)
        else:
            prospensity = p
        
        Dhat_0 = []
        Dhat_1 = []
        cnt0 = cnt1= 0
        for i in range(len(ytest)):
            if treatment[i]==0:
                Dhat_0.append(D0[cnt0])
                cnt0+=1
                Dhat_1.append(prob1[i])
            if treatment[i]==1:
                Dhat_1.append(D1[cnt1])
                cnt1+=1
                Dhat_0.append(prob0[i])
  
        return prospensity*np.array(Dhat_0)+(1-prospensity)*np.array(Dhat_1) 
