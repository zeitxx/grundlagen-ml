
# coding: utf-8

# ### Grundlagen des maschinellen Lernens
# 
# *Einstieg in die Klassifikation*
# 
# Dr. Christian Herff

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#get_ipython().magic('matplotlib inline')

digits = datasets.load_digits()
iris = datasets.load_iris()




def euclidNorm(v):    
    return np.sqrt(np.sum((v)**2))

def maxNorm(v):
    return np.max(np.abs(v))

def sumNorm(v):
    return np.sum(np.abs(v))

class KNNClassifier:
    data = []
    label= []
    norm=[]
    k=[]
    
    def __init__(self, k=5,norm=euclidNorm):
        self.k=k
        self.norm=norm
    
    def fit(self,x,y):
        self.data=x
        self.label=y
        
    def predict(self,x):
        predictedLabel=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            dists=np.zeros(self.data.shape[0])
            for j in range(self.data.shape[0]):
                dists[j]=self.norm(x[i,:]-self.data[j,:])
            nn=np.argsort(dists)[:self.k]
            lbl=self.label[nn]
            indLbl,occs=np.unique(lbl,return_counts=True)
            predictedLabel[i]=indLbl[np.argmax(occs)]
        return predictedLabel

def hyperCube(v):
    if np.max(np.abs(v))<.5:
        return 1
    else:
        return 0

def hyperSphere(v):
    if np.sqrt(np.sum((v)**2))<1:
        return 1
    else:
        return 0

class ParzenWindowClassifier:
    data = []
    label= []
    window=[]
    h=[]
    
    def __init__(self, window=hyperCube, h=1):
        self.window=window
        self.h=h
        
    def fit(self,x,y):
        self.data=x
        self.label=y
        
    def predict(self,x):
        predictedLabel=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            likeli=np.zeros(np.max(self.label)+1)
            for j in range(self.data.shape[0]):
                likeli[self.label[j]]+=1/float(self.h) * self.window((self.data[j,:]-x[i,:])/float(self.h))
            #print(np.sum(likeli))
            likeli=likeli/float(self.data.shape[0])
            #print(likeli)
            predictedLabel[i]=np.argmax(likeli)
        return predictedLabel

def accuracy(prediction, reference):
    correct = np.sum(prediction==reference)   
    return float(correct)/len(reference)

def getPrior(reference):
    classes,occs=np.unique(reference,return_counts=True)
    return float(np.max(occs))/len(reference)

def confusionMatrix(prediction, reference):
    classes=np.unique(reference)
    matrix = np.zeros((len(classes),len(classes)))
    for s in range(len(reference)):
        matrix[reference[s],prediction[s]]+=1
    return matrix

def precision(prediction,reference,label):
    tp=np.sum(prediction[reference==label]==
              reference[reference==label])
    fp=np.sum(prediction[reference!=label]==label)
    return float(tp)/(tp+fp)

def recall(prediction,reference,label):
    tp=np.sum(prediction[reference==label]==
              reference[reference==label])
    fn=len(reference[reference==label])-tp
    return float(tp)/(tp+fn)

def fscore(prediction,reference,label):
    prec=precision(prediction,reference,label)
    rec=recall(prediction,reference,label)
    return 2*(prec*rec)/(prec+rec)

def uar(prediction,reference):
    classes=np.unique(reference)
    recalls=np.zeros(len(classes))
    for i,j in enumerate(classes):
        recalls[i]=recall(prediction,reference,j)
    return np.mean(recalls)

def kfold(numSamples,k=10,shuffle=False):
    folds=[]
    idx=list(range(numSamples))
    if shuffle:
        np.random.shuffle(idx)
    step=numSamples/k
    for j in range(k):
        test=idx[int(j*step):int((j+1)*step)]
        train=idx[:int(j*step)] + idx[int((j+1)*step):]
        folds.append((train,test))
    return folds

def pairedTTestRanker(data,label):
    featurePs=np.zeros(data.shape[1])
    for f in range(data.shape[1]):
        pVals=[]
        classes=np.unique(label)
        for i in range(len(classes)):
            for j in range(i,len(classes)):
                t, prob = ttest_ind(data[label==classes[i],f],data[label==classes[j],f])
                pVals.append(prob)
        featurePs[f]=np.mean(pVals)
    return np.argsort(featurePs)

def fisherRatioRanker(data,label):
    featureFR=np.zeros(data.shape[1])
    for f in range(data.shape[1]):
        fisherRs=[]
        classes=np.unique(label)
        for i in range(len(classes)):
            for j in range(i,len(classes)):
                x=np.mean(data[label==classes[i],f])
                y=np.mean(data[label==classes[j],f])
                s1=np.var(data[label==classes[i],f])
                s2=np.var(data[label==classes[j],f])
                fr = ((x-y)**2) / (s1+s2)
                fisherRs.append(fr)
        featureFR[f]=np.mean(fisherRs)
    return np.argsort(featureFR)[::-1]

def wrapperRanker(data,label, num2select, est, nestedK): 
    features=list(range(data.shape[1]))
    selected=[]
    for n in range(num2select):
        featureAcc=np.zeros(len(features))
        for i,f in enumerate(features):
            accs=[]
            for train,test in kfold(len(label),nestedK):
                est.fit(data[train,:][:,selected+[f]],label[train])
                prediction=est.predict(data[test,:][:,selected+[f]])
                accs.append(accuracy(prediction,label[test]))
            featureAcc[i]=np.mean(accs)
        selected.append(features[np.argmax(featureAcc)])
        del features[np.argmax(featureAcc)]
    return selected

def pca(data):
    # Calculate Covariance Matrix
    cov=np.cov(data.T)
    # Calculate Eigenvalues and Eigenvectors
    w, v = np.linalg.eig(cov)
    # Sort them
    s= np.argsort(w)[::-1]
    return v[:,s]

class KMeans:
    k=[]
    norm=[]
    def __init__(self, k=5,norm=euclidNorm):
        self.k=k
        self.norm=norm
    def cluster(self,x):
        means=np.zeros((self.k,x.shape[1]))
        meanIdx=np.random.choice(x.shape[0],self.k,replace=False)
        newMeans=x[meanIdx,:]
        i=0
        knn=KNNClassifier(k=1,norm=self.norm)
        while (not (means==newMeans).all()) and i<10:
            means=newMeans
            knn.fit(means,np.arange(self.k))
            label=knn.predict(x)
            newMeans=np.zeros((self.k,x.shape[1]))
            for c in range(self.k):
                newMeans[c,:]=np.mean(x[label==c,:],axis=0)
            i+=1
        return label

def uniGaussian(x, mu, sigma):
    normTerm = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    meanDiff = x - mu
    expTerm = np.exp(-meanDiff**2/(2*sigma**2))
    return normTerm * expTerm

class NaiveBayes:
    classLabels=[]
    priors=[]
    means=[]
    sigmas=[]
    
    def __init__(self):
            pass
        
    def fit(self, x, y):
        self.classLabels, occs =np.unique(y,return_counts=True)
        self.priors=occs/len(y)
        self.means=np.zeros((self.classLabels.shape[0],x.shape[1]))
        self.sigmas=np.zeros((self.classLabels.shape[0],x.shape[1]))
        for f in range(x.shape[1]):
            for i,c in enumerate(self.classLabels):
                self.means[i,f]=np.mean(x[y==c,f])
                self.sigmas[i,f]=np.std(x[y==c,f])
                
    def predict(self,x):
        labels=[]
        for i in range(x.shape[0]):
            ps=np.zeros(self.means.shape)
            for f in range(x.shape[1]):
                for c in range(ps.shape[0]):
                    ps[c,f]=uniGaussian(x[i,f],self.means[c,f],self.sigmas[c,f])
            
            ps=self.priors*np.prod(ps,axis=1)
            ps=ps/np.sum(ps) #Normalisierung für echte Wahrscheinlichkeiten
            labels.append(self.classLabels[np.argmax(ps)])
        return labels  

def gaussian(x, mean, covariance):
    D = mean.shape[0]
    normTerm = 1.0 / np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(covariance))
    meanDiff = x - mean
    expTerm = np.exp(-0.5 * np.dot(np.dot(meanDiff.transpose(), np.linalg.pinv(covariance)), meanDiff))
    return normTerm * expTerm

class GaussianBayes:
    classLabels=[]
    priors=[]
    means=[]
    covs=[]
    
    def __init__(self):
            pass
        
    def fit(self, x, y):
        self.classLabels, occs =np.unique(y,return_counts=True)
        self.priors=occs/len(y)
        self.means=np.zeros((self.classLabels.shape[0],x.shape[1]))
        self.covs=np.zeros((self.classLabels.shape[0],x.shape[1],x.shape[1]))
        for i,c in enumerate(self.classLabels):
            self.means[i,:]=np.mean(x[y==c,:],axis=0)
            self.covs[i,:,:]=np.cov(x[y==c,:].T)
                
    def predict(self,x):
        labels=[]
        for i in range(x.shape[0]):
            ps=np.zeros(len(self.classLabels))
            for c in range(ps.shape[0]):
                ps[c]=gaussian(x[i,:],self.means[c,:],self.covs[c,:,:])
            ps=self.priors*ps
            ps=ps/np.sum(ps) #Normalisierung für echte Wahrscheinlichkeiten
            labels.append(self.classLabels[np.argmax(ps)])
        return labels    

class GaussianMixtureModel:
    means=[]
    covs=[]
    cweights=[]
    k=[]
    def __init__(self,k=3):
        self.k=k
        
    def fit(self,x, max_iters = 100):
        means=np.zeros((self.k,x.shape[1]))
        meanIdx=np.random.choice(x.shape[0],self.k,replace=False)
        newMeans=x[meanIdx,:]
        covs= [np.cov(x.T)] * self.k
        w = [1./self.k] * self.k
        membership = np.zeros((x.shape[0], self.k))
        i=0
        finfo=np.finfo('float')
        while (not (means==newMeans).all()) and i<max_iters:
            means=newMeans
            #Expectation
            for k in range(self.k):
                membership[:, k] = [w[k] *gaussian(x[s,:],means[k,:], covs[k]) for s in range(x.shape[0])]         
            if np.min(membership)==0:
                membership+=finfo.tiny
            membership = membership/np.sum(membership,axis=1)[:,None]
            #Maximization 
            newMeans=np.zeros((self.k,x.shape[1]))
            
            for k in range(self.k):
                w[k]=np.sum(membership[:,k]/np.sum(membership))
                newMeans[k,:]=np.sum(x*membership[:,k][:,None],axis=0)/np.sum(membership[:,k])
                covs[k] = sum([membership[s,k]*np.outer(x[s,:]-newMeans[k,:], x[s,:]-newMeans[k,:]) for s in range(x.shape[0])]) / np.sum(membership[:,k])
            i+=1
            self.means=newMeans
            self.covs=covs
            self.cweights=w
            #plotIrisGMM(self,i)
        
    def predict(self,x):
        likelis=np.zeros(x.shape[0])
        for s in range(x.shape[0]):
            ps=np.zeros(self.k)
            for c in range(self.k):
                ps[c]=self.cweights[c] * gaussian(x[s,:],self.means[c,:],self.covs[c])
            likelis[s]=np.sum(ps)
        return likelis

class GMMBayes:
    classLabels=[]
    priors=[]
    gmms=[]
    k=[]
    
    def __init__(self,k=2):
        self.k=k
        
    def fit(self, x, y):
        self.classLabels, occs =np.unique(y,return_counts=True)
        self.priors=occs/len(y)
        self.gmms=[]
        for i,c in enumerate(self.classLabels):
            gmm=GaussianMixtureModel(k=self.k)
            gmm.fit(x[y==c,:])
            self.gmms.append(gmm)
            
    def predict(self,x):
        labels=[]
        ps=np.zeros((len(self.classLabels),x.shape[0]))
        for c in range(ps.shape[0]):
            ps[c,:]=self.gmms[c].predict(x)
        ps=self.priors[:,None]*ps
        
        labels.append(self.classLabels[np.argmax(ps,axis=0)])
        return labels

def gini(groups):
    classLabels=np.unique([item for sublist in groups for item in sublist])
    props=np.zeros((len(groups),len(classLabels)))
    for i,g in enumerate(groups):
        for j,l in enumerate(classLabels):
            props[i,j]=np.sum(g==l)/len(g)
    return np.sum(props * (1-props))

def evalSplits(x,y):
    ginis=np.ones(x.shape)
    for s in range(x.shape[0]):
        for f in range(x.shape[1]):
            left=y[x[:,f]<x[s,f]]
            right=y[x[:,f]>=x[s,f]]
            if len(left)>0 and len(right)>0:
                ginis[s,f] = gini([left,right])
    pos = np.unravel_index(ginis.argmin(), ginis.shape)
    #print(y[x[:,pos[1]]<x[pos]])
    left = np.argwhere(x[:,pos[1]]<x[pos]).ravel()
    right = np.argwhere(x[:,pos[1]]>=x[pos]).ravel()
    return (pos[1],x[pos],left,right)

class ClassificationTree:
    max_depth=[]
    tree=[]
    
    def __init__(self, max_depth=3):
        self.max_depth=max_depth
        
    def buildNode(self,x,y,depth):
        f, val, lIx, rIx = evalSplits(x,y)
        nodes=[[],[]]
        for i, nIx in enumerate([lIx,rIx]):
            if len(np.unique(y[nIx]))>1 and depth<self.max_depth:
                nodes[i] = self.buildNode(x[nIx,:],y[nIx],depth+1)
            else:
                lbl, occs = np.unique(y[nIx],return_counts=True)
                nodes[i]=lbl[np.argmax(occs)]
        return ((f, val),(nodes[0],nodes[1]))
    
    def fit(self, x, y):
        self.tree = self.buildNode(x,y,1)
        
    def predict(self,x):
        labels=[]
        for s in range(x.shape[0]):
            node=self.tree
            while isinstance(node, tuple):
                if x[s,node[0][0]]<node[0][1]:
                    node=node[1][0]
                else:
                    node=node[1][1]
            labels.append(node)
        return labels

class LDA:
    classLabels=[]
    theta=[]
    b=[]
    
    def __init__(self):
        pass
        
    def fit(self, x, y):
        self.classLabels =np.unique(y)
        if len(self.classLabels)>2:
            raise Exception('LDA only works for 2 classes')
        mus=np.zeros((2,x.shape[1]))
        covs=np.zeros((2,x.shape[1],x.shape[1]))
        for i,c in enumerate(self.classLabels):
            mus[i,:]=np.mean(x[y==c,:],axis=0)
            covs[i,:,:]=np.cov(x[y==c,:].T)
        self.theta=np.dot(np.linalg.inv(covs[0,:,:]+covs[1,:,:]),(mus[1,:]-mus[0,:]))
        self.b=-np.dot(self.theta.T, (mus[0,:]+mus[1,:]))/2   
            
    def predict(self,x):
        labels=[]
        for i in range(x.shape[0]):
            y=np.sign(np.dot(self.theta,x[i,:])+self.b)
            y=1 if y==1 else 0
            labels.append(self.classLabels[y])
        return labels 

class ShrinkageLDA:
    shrinkage=[]
    classLabels=[]
    theta=[]
    b=[]
    
    def __init__(self,shrinkage=.3):
        self.shrinkage=shrinkage
        
    def fit(self, x, y):
        self.classLabels =np.unique(y)
        if len(self.classLabels)>2:
            raise Exception('LDA only works for 2 classes')
        mus=np.zeros((2,x.shape[1]))
        covs=np.zeros((2,x.shape[1],x.shape[1]))
        for i,c in enumerate(self.classLabels):
            mus[i,:]=np.mean(x[y==c,:],axis=0)
            covs[i,:,:]=np.cov(x[y==c,:].T) 
            covs[i,:,:] = (1-self.shrinkage) *  covs[i,:,:] + self.shrinkage * np.eye(x.shape[1]) * covs[i,:,:]
        self.theta=np.dot(np.linalg.pinv(covs[0,:,:]+covs[1,:,:]),(mus[1,:]-mus[0,:]))
        self.b=-np.dot(self.theta.T, (mus[0,:]+mus[1,:]))/2   
            
    def predict(self,x):
        labels=[]
        for i in range(x.shape[0]):
            y=np.sign(np.dot(self.theta,x[i,:])+self.b)
            y=1 if y==1 else 0
            labels.append(self.classLabels[y])
        return labels

class OneVsOneClassifier:
    classLabels=[]
    classifier=[]
    trainedClassifiers=[]
    
    def __init__(self, classifier=LDA):
        self.classifier=classifier
        
    def fit(self, x,y):
        self.classLabels=np.unique(y)
        self.trainedClassifiers=[]
        for i in range(len(self.classLabels)):
            self.trainedClassifiers.append([])
            for j in range(i+1,len(self.classLabels)):
                est=self.classifier()
                selected=np.array([a in [self.classLabels[i],self.classLabels[j]] for a in y])
                est.fit(x[selected,:],y[selected])
                self.trainedClassifiers[i].append(est)
                
    def predict(self,x):
        labels=[]
        votes=np.zeros((x.shape[0],len(self.classLabels)))
        for i in range(len(self.classLabels)):
            for j,_ in enumerate(range(i+1,len(self.classLabels))):
                predicted=self.trainedClassifiers[i][j].predict(x)
                for l in range(len(predicted)):
                    votes[l,np.where(self.classLabels==predicted[l])]+=1
        labels=[self.classLabels[np.argmax(votes,axis=1)]]
        return labels

def rmse(predicted,target):
    return np.sqrt(np.mean((predicted.ravel()-target)**2))

class LinearRegression:
    a=[]
    b=[]
    
    def __init__(self):
        pass
    
    def fit(self,x,y):
        cov=np.cov(np.concatenate((x,y),axis=1).T)
        self.a=np.dot(cov[:x.shape[1],x.shape[1]:].T,np.linalg.pinv(cov[:x.shape[1],:x.shape[1]])) 
        self.b=np.mean(y,axis=0) - np.dot(self.a, np.mean(x,axis=0))
        
    def predict(self,x):
        out = np.dot(self.a,x.T) + self.b[:,None]
        return out.T

class KNNRegression:
    data = []
    target= []
    norm=[]
    k=[]
    
    def __init__(self, k=5,norm=euclidNorm):
        self.k=k
        self.norm=norm
    
    def fit(self,x,y):
        self.data=x
        self.target=y
        
    def predict(self,x):
        predicted=np.zeros((x.shape[0],self.target.shape[1]))
        for i in range(x.shape[0]):
            dists=np.zeros(self.data.shape[0])
            for j in range(self.data.shape[0]):
                dists[j]=self.norm(x[i,:]-self.data[j,:])
            nn=np.argsort(dists)[:self.k]
            trgt=np.mean(self.target[nn],axis=0)
            predicted[i]=trgt
        return predicted

def evalRegressionSplits(x,y,min_leafs):
    allVs=np.zeros(x.shape)+np.inf
    totalV=np.var(y)/(len(y)**2)
    for s in range(x.shape[0]):
        for f in range(x.shape[1]):
            left=np.argwhere(x[:,f]<x[s,f])
            right=np.argwhere(x[:,f]>=x[s,f])
            if len(left)>=min_leafs and len(right)>=min_leafs:
                allVs[s,f]=np.abs(totalV-(np.var(y[left])/len(left)**2+np.var(y[right])/len(right)**2))
    pos = np.unravel_index(allVs.argmin(), allVs.shape)
    left = np.argwhere(x[:,pos[1]]<x[pos]).ravel()
    right = np.argwhere(x[:,pos[1]]>=x[pos]).ravel()
    return (pos[1],x[pos],left,right)

class RegressionTree:
    max_depth=[]
    min_leafs=[]
    tree=[]
    
    def __init__(self, max_depth=3,min_leafs=10):
        self.max_depth=max_depth
        self.min_leafs=min_leafs
        
    def buildNode(self,x,y,depth):
        f, val, lIx, rIx = evalRegressionSplits(x,y,self.min_leafs)
        nodes=[[],[]]
        for i, nIx in enumerate([lIx,rIx]):
            if len(nIx)>self.min_leafs*2 and depth<self.max_depth:
                nodes[i] = self.buildNode(x[nIx,:],y[nIx],depth+1)
            else:
                nodes[i]=np.mean(y[nIx],axis=0)
        return ((f, val),(nodes[0],nodes[1]))
    
    def fit(self, x, y):
        self.tree = self.buildNode(x,y,1)
        
    def predict(self,x):
        labels=[]
        for s in range(x.shape[0]):
            node=self.tree
            while isinstance(node, tuple):
                if x[s,node[0][0]]<node[0][1]:
                    node=node[1][0]
                else:
                    node=node[1][1]
            labels.append(node)
        return labels

class BaggingClassifier:
    num_est=[]
    subsample=[]
    est=[]
    
    classLabels=[]
    trained_est=[]
    
    def __init__(self, est=LDA,num_est=50, subsample=0.8):
        self.est=est
        self.num_est=num_est
        self.subsample=subsample
        
    def fit(self,x,y):
        self.classLabels=np.unique(y)
        self.trained_est=[]
        for t in range(self.num_est):
            est = self.est()
            selected=np.random.choice(x.shape[0],int(x.shape[0]*self.subsample),replace=False)
            est.fit(x[selected,:],y[selected])
            self.trained_est.append(est)
            
    def predict(self,x):
        allPred= np.zeros((x.shape[0],len(self.classLabels)))
        for t in self.trained_est:
            prediction = t.predict(x)
            for s in range(len(prediction)):
                allPred[s,np.argwhere(self.classLabels==prediction[s])]+=1
        out = [self.classLabels[np.argmax(allPred[s,:])] for s in range(allPred.shape[0])]
        return np.array(out)

class RandomTree:
    max_depth=[]
    tree=[]
    
    def __init__(self, max_depth=3):
        self.max_depth=max_depth
        
    def buildNode(self,x,y,depth):
        feat=np.random.choice(range(x.shape[1]))
        f, val, lIx, rIx = evalSplits(x[:,feat][:,None],y)
        f=feat
        nodes=[[],[]]
        for i, nIx in enumerate([lIx,rIx]):
            if len(np.unique(y[nIx]))>1 and depth<self.max_depth:
                nodes[i] = self.buildNode(x[nIx,:],y[nIx],depth+1)
            else:
                lbl, occs = np.unique(y[nIx],return_counts=True)
                nodes[i]=lbl[np.argmax(occs)]
        return ((f, val),(nodes[0],nodes[1]))
    
    def fit(self, x, y):
        self.tree = self.buildNode(x,y,1)
        
    def predict(self,x):
        labels=[]
        for s in range(x.shape[0]):
            node=self.tree
            while isinstance(node, tuple):
                if x[s,node[0][0]]<node[0][1]:
                    node=node[1][0]
                else:
                    node=node[1][1]
            labels.append(node)
        return labels

class RandomForest:
    max_depth=[]
    num_trees=[]
    subsample=[]
    
    classLabels=[]
    trees=[]
    
    def __init__(self, max_depth=3,num_trees=50, subsample=0.8):
        self.max_depth=max_depth
        self.num_trees=num_trees
        self.subsample=subsample
        
    def fit(self,x,y):
        self.classLabels=np.unique(y)
        self.trees=[]
        for t in range(self.num_trees):
            rt = RandomTree(max_depth=self.max_depth)
            selected=np.random.choice(x.shape[0],int(x.shape[0]*self.subsample),replace=False)
            rt.fit(x[selected,:],y[selected])
            self.trees.append(rt)
            
    def predict(self,x):
        allPred= np.zeros((x.shape[0],len(self.classLabels)))
        for t in self.trees:
            prediction = t.predict(x)
            for s in range(len(prediction)):
                allPred[s,np.argwhere(self.classLabels==prediction[s])]+=1
        out = [self.classLabels[np.argmax(allPred[s,:])] for s in range(allPred.shape[0])]
        return np.array(out)

class ExtremeRandomTree:
    max_depth=[]
    tree=[]
    
    def __init__(self, max_depth=3):
        self.max_depth=max_depth
        
    def buildNode(self,x,y,depth):
        f=np.random.choice(range(x.shape[1]))
        sample=np.random.choice(range(x.shape[0]))
        val=x[sample,f]
        lIx = np.argwhere(x[:,f]<val).ravel()
        rIx = np.argwhere(x[:,f]>=val).ravel()
        while (len(lIx)==0 or len(rIx)==0):
            f=np.random.choice(range(x.shape[1]))
            sample=np.random.choice(range(x.shape[0]))
            val=x[sample,f]
            lIx = np.argwhere(x[:,f]<val).ravel()
            rIx = np.argwhere(x[:,f]>=val).ravel()
        nodes=[[],[]]
        for i, nIx in enumerate([lIx,rIx]):
            if len(np.unique(y[nIx]))>1 and depth<self.max_depth:
                nodes[i] = self.buildNode(x[nIx,:],y[nIx],depth+1)
            else:
                lbl, occs = np.unique(y[nIx],return_counts=True)
                nodes[i]=lbl[np.argmax(occs)]
        return ((f, val),(nodes[0],nodes[1]))
    
    def fit(self, x, y):
        self.tree = self.buildNode(x,y,1)
        
    def predict(self,x):
        labels=[]
        for s in range(x.shape[0]):
            node=self.tree
            while isinstance(node, tuple):
                if x[s,node[0][0]]<node[0][1]:
                    node=node[1][0]
                else:
                    node=node[1][1]
            labels.append(node)
        return labels

class ExtremeForest:
    max_depth=[]
    num_trees=[]
    subsample=[]
    classLabels=[]
    trees=[]
    
    def __init__(self, max_depth=3,num_trees=150,subsample=0.8):
        self.max_depth=max_depth
        self.num_trees=num_trees
        self.subsample=subsample
        
    def fit(self,x,y):
        self.classLabels=np.unique(y)
        self.trees=[]
        for t in range(self.num_trees):
            selected=np.random.choice(x.shape[0],int(x.shape[0]*self.subsample),replace=False)
            rt = ExtremeRandomTree(max_depth=self.max_depth)
            rt.fit(x[selected,:],y[selected])
            self.trees.append(rt)
            
    def predict(self,x):
        allPred= np.zeros((x.shape[0],len(self.classLabels)))
        for t in self.trees:
            prediction = t.predict(x)
            for s in range(len(prediction)):
                allPred[s,np.argwhere(self.classLabels==prediction[s])]+=1
        out = [self.classLabels[np.argmax(allPred[s,:])] for s in range(allPred.shape[0])]
        return np.array(out)  

def classificationError(groups, y, weights):
    error = 0
    classLabel=[]
    for g in groups:
        lbl, occs = np.unique(y[g], return_counts=True)
        error += np.sum((y[g] != lbl[np.argmax(occs)]) * weights[g])
        classLabel.append(lbl[np.argmax(occs)])
    return error,classLabel

class DecisionStump:
    feat=[]
    val=[]
    classLabel=[]
    
    def __init__(self):
        pass
    
    def fit(self,x,y,weights=None):
        if weights is None:
            weights=np.ones(len(y))/len(y)
        mC=np.zeros(x.shape)+np.inf
        for s in range(x.shape[0]):
            for f in range(x.shape[1]):
                left=x[:,f]<x[s,f]
                right=x[:,f]>=x[s,f]
                if np.sum(left)>0 and np.sum(right)>0:
                    mC[s,f],_=classificationError([left,right],y, weights)
        pos = np.unravel_index(mC.argmin(), mC.shape)
        self.feat=pos[1]
        self.val=x[pos]
        left=x[:,self.feat]<self.val
        right=x[:,self.feat]>=self.val
        _,self.classLabel=classificationError([left,right],y,weights)
        
    def predict(self,x):
        label=[]
        for s in range(x.shape[0]):
            if x[s,self.feat]<self.val:
                label.append(self.classLabel[0])
            else:
                label.append(self.classLabel[1])
        return label

class AdaBoost:
    weak_learner=[]
    
    classLabels=[]
    trained_learner=[]
    alphas=[]
    num_learner=[]
    
    def __init__(self,weak_learner=DecisionStump, num_learner=50):
        self.weak_learner=weak_learner
        self.num_learner=num_learner
        
    def fit(self,x,y):
        self.classLabels=np.unique(y)
        self.trained_learner=[]
        self.alphas=[]
        weights=np.ones(len(y))/len(y)
        for t in range(self.num_learner):
            l = self.weak_learner()
            l.fit(x,y,weights)
            prediction=l.predict(x)
            epsilon = np.sum((prediction!=y)*weights)
            if epsilon>0:
                alpha=0.5 * np.log((1-epsilon)/epsilon)
            else:
                alpha=1
            self.trained_learner.append(l)
            self.alphas.append(alpha)
            t=(prediction==y)
            t=np.array([1 if a else -1 for a in t])
            weights=weights * np.exp(- alpha * t)
            weights = weights /np.sum(weights)
            
    def predict(self, x):
        prediction=np.zeros((x.shape[0],2))
        for l, est in enumerate(self.trained_learner):
            weak_out = est.predict(x)
            for s in range(len(weak_out)):
                prediction[s,np.argwhere(self.classLabels==
                                         weak_out[s])]+=self.alphas[l]
        out = [self.classLabels[np.argmax(prediction[s,:])] for s in 
               range(prediction.shape[0])]
        return np.array(out)

class Perceptron:
    max_iters=[]
    classLabels=[]
    w = []
    
    def __init__(self,max_iters=1):
        self.max_iters=max_iters
    
    def fit(self,x,y):
        self.classLabels=np.unique(y)
        if len(self.classLabels)>2:
            raise('Perceptron only works for two classes')
        self.w=np.zeros(x.shape[1]+1)
        y=((y==self.classLabels[1])*1)
        for iter in range(self.max_iters):
            shuffle=np.arange(x.shape[0])
            np.random.shuffle(shuffle)
            x=x[shuffle,:]
            y=y[shuffle]
            for s in range(x.shape[0]):
                inS = np.hstack((x[s,:],1))
                self.w = self.w + (y[s]-int( np.dot(self.w,inS)>0 )) *inS
                
    def predict(self,x):
        label=[]
        for s in range(x.shape[0]):
            inS = np.hstack((x[s,:],1))
            out=int( np.dot(self.w,inS)>0 )
            label.append(self.classLabels[out])
        return label

def encodeOneHot(y):
    classLabel=np.unique(y)
    ys=np.zeros((y.shape[0],len(classLabel)))
    for s in range(len(y)):
        ys[s,np.argwhere(classLabel==y[s])[0]]=1
    return ys

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

def relu(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def shuffle(x,y):
    shuffle=np.arange(x.shape[0])
    np.random.shuffle(shuffle)
    x=x[shuffle,:]
    y=y[shuffle,:]
    return x,y

class SimpleNN():
    hidden=[]
    eta=[]   
    w1=[]
    w2=[]
    
    def __init__(self,hidden_nodes=100,eta=1e-4):
        self.hidden=hidden_nodes
        self.eta=eta
        
    def fit(self,x,y,max_iters=100):
        self.classLabel=np.unique(y)
        y=encodeOneHot(y)
        x = np.hstack((x,np.ones((x.shape[0],1))))   
        self.w1 = np.random.randn(x.shape[1], self.hidden)
        self.w2 = np.random.randn(self.hidden, len(self.classLabel))
        errs=[]
        for iter in range(max_iters):
            x,y=shuffle(x,y)
            for s in range(x.shape[0]):
                y_pred,h =self.forward(x[s,:])
                err=(y[s,:]-y_pred)
                errs.append(np.sum(err**2))
                grad_w2=h[:,None].dot(err[:,None].T)
                grad_hidden=err.dot(self.w2.T)
                grad_hidden=relu(grad_hidden)
                grad_w1=x[s,:][:,None].dot(grad_hidden[:,None].T)
                self.w2+=self.eta*grad_w2
                self.w1+=self.eta*grad_w1
        return(errs)
    
    def forward(self,x):
        h = x.dot(self.w1)
        h = relu(h)
        y_pred = h.dot(self.w2)
        y_pred = softmax(y_pred)
        return y_pred,h
        
    def predict(self,x):
        x = np.hstack((x,np.ones((x.shape[0],1))))
        out,_=self.forward(x)
        labels=[self.classLabel[np.argmax(out,axis=1)]]
        return labels
