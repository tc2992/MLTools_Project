from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random


class HonestRF():
    def __init__(self,rf=RandomForestClassifier()):
        self.rf = rf
        self.leaf = None
        self.proba = None #(n_nodes, n_trees)
    
    def fit(self,X,y):
        idx=list(np.arange(len(y)))
        random.shuffle(idx)
        train_idx = idx[:len(y)//2]
        test_idx = idx[len(y)//2:]
    
        Xtrain=X[train_idx]
        ytrain=y[train_idx]
        Xtest=X[test_idx]
        ytest=y[test_idx]
    
        self.rf = self.rf.fit(Xtrain,ytrain)
        self.set_proba(Xtest,ytest)
        return self

    def set_proba(self,X,y):
        mat = self.rf.apply(X)
        self.leaf = np.unique(mat)
        self.proba = np.zeros((max(self.leaf)+1,mat.shape[1]))
        for i in self.leaf:
            for j in range(mat.shape[1]):
                tree = mat[:,j]
                idx = np.where(tree==i)[0]
                if len(idx)==0:
                    continue
                prob1 = sum(y[idx])/len(y[idx])
                self.proba[i,j] = prob1
        
    def predict_proba(self,X):
        mat = self.rf.apply(X)
        proba = np.zeros((X.shape[0],2))
        for i in range(X.shape[0]):
            idx = mat[i]
            prob1 = 0
            for j in range(mat.shape[1]):
                if idx[j] >= self.proba.shape[0]:
                    continue
                prob1 += self.proba[idx[j],j]
            prob1 /= mat.shape[1]
            prob0=1-prob1
            proba[i,0] = prob0
            proba[i,1] = prob1
        return proba
        
        
class BaseTClassifier(object):
    def __init__(self,learner0,learner1):
        self.learner0 = learner0
        self.learner1 = learner1
        self.model0 = None
        self.model1 = None
    
    def fit(self,X,y,treatment):
        X0 = X[treatment==0]
        X1 = X[treatment==1]
        y0 = y[treatment==0]
        y1 = y[treatment==1]
    
        self.model0 = self.learner0.fit(X0, y0)
        self.model1 = self.learner1.fit(X1, y1)
    
    def predict(self,X, idx):
        if idx==0:
            prob = self.model0.predict_proba(X)[:, 1]
        else:
            prob = self.model1.predict_proba(X)[:, 1]
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



