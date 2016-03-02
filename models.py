'''
Created on 29 Dec 2015

@author: af
'''
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression, LassoCV
import numpy as np
from sklearn.grid_search import GridSearchCV
from collections import defaultdict, Counter
import codecs
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
import subprocess
import logging
import pdb
import random
import lda
import pickle
import os
from scipy import sparse
from scipy.stats import entropy
import operator
import itertools
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
import sys
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
 
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def sgdclassifier(training_samples, eval_samples, vectorizer, do_grid_search=True):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    #clf = SGDClassifier(loss='log', penalty= 'l2',l1_ratio=0.0, n_iter=30, shuffle=True, verbose=False, 
    #                    n_jobs=4, alpha=1e-4, average=True, class_weight=None)
    clf = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, 
                             max_iter=50, multi_class='ovr', verbose=0, warm_start=False, n_jobs=10)
    print clf
    if do_grid_search:
        tuned_parameters = {'C': [0.0001, 0.001, 0.01, 0.05, 0.07, 0.1, 0.15, 0.2],
                            #'penalty':['l1', 'l2']
                            }
        clf = GridSearchCV(clf, tuned_parameters, cv=5, scoring='log_loss', n_jobs=10)
    ''' best params
    {'penalty': 'l2', 'alpha': 0.0001, 'average': True, 'class_weight': None}
    '''
    #clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=clf.best_params_['alpha'], l1_ratio=0.15, fit_intercept=True, n_iter=10, shuffle=False, verbose=True)
    clf.fit(X_train, Y_train)
    #y_train_true, y_train_pred = Y_train, clf.predict(X_train)
    print_top_10_words = True
    
    if do_grid_search:
        if print_top_10_words:
            feature_names = np.asarray(vectorizer.get_feature_names())
            print("top 10 keywords per class:")
            for i in range(0, clf.best_estimator_.coef_.shape[0]):
                    top10 = np.argsort(clf.best_estimator_.coef_[i])[-50:]
                    #print(" ,".join(feature_names[top10]))
                    #bottom10 = np.argsort(clf.best_estimator_.coef_[i])[0:50]
                    #print(" ,".join(feature_names[bottom10]))
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
    else:
        pass
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)

    print(clf)
    #scores = cross_validation.cross_val_score(clf.best_estimator_, X_train, Y_train, cv=10, scoring='log_loss')
    #print scores, np.mean(scores), np.median(scores)
    y_true, y_pred = Y_eval, clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)
    #print(classification_report(y_true, y_pred))
    #print()
    #print 'acc train: ', accuracy_score(Y_train, y_train_pred, normalize=True, sample_weight=None), 'f1 train: ', f1_score(Y_train, y_train_pred, average='macro')
    return  y_pred, y_prob
def rfclassifier(training_samples, eval_samples, do_grid_search=False):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    
    clf = RandomForestClassifier(n_estimators=4000, criterion='gini', max_depth=None, 
                                 min_samples_split=64, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                 max_features=320, max_leaf_nodes=None, bootstrap=True, oob_score=True, 
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)

    if do_grid_search:
        to_be_tuned_parameters = {
                                  #'n_estimators':[500, 2000, 4000],
                                  'max_features':[200, 320, 500],
                                  'min_samples_split':[50, 64, 128],
                                  #'min_samples_leaf': [1, 2],
                                  }
        clf = GridSearchCV(clf, to_be_tuned_parameters, cv=5, n_jobs=10, scoring='log_loss')

    #Best parameters set found on development set:
    #()
    #{'max_features': 'log2', 'min_samples_split': 8, 'criterion': 'gini', 'min_samples_leaf': 1}
    
                

    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        print("Best parameters set found on development set:")
        print()
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)
    Y_eval = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)
    if do_grid_search:
        feature_importance = clf.best_estimator_.feature_importances_
    else:
        feature_importance = clf.feature_importances_
    return Y_eval, Y_prob, feature_importance

def etclassifier(training_samples, eval_samples, do_grid_search=True):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples

        
    clf = ExtraTreesClassifier(max_depth=None, n_estimators=1000,
                                 min_weight_fraction_leaf=0.0, max_features=None, min_samples_split=16, criterion='gini',
                                 min_samples_leaf=2, max_leaf_nodes=None, oob_score=False, bootstrap=True,
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)
    to_be_tuned_parameters = {
                              #'n_estimators':[500, 2000, 4000],
                              'max_features':['log2', 'auto', None],
                              'min_samples_split':[2, 8, 16],
                              'min_samples_leaf': [1, 2],

                            }
    if do_grid_search:
        clf = GridSearchCV(clf, to_be_tuned_parameters, cv=5, n_jobs=5, scoring='log_loss')
    #Best parameters set found on development set:
    #()
    #{'max_features': None, 'min_samples_split': 10, 'n_estimators': 1000, 'min_samples_leaf': 2}
    
    
                

    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        print("Best parameters set found on development set:")
        print()
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)
    Y_eval = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)
    return Y_eval, Y_prob
 

 
 
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def xgboostclassifier(training_samples, eval_samples, do_grid_search=False):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples

    clf = XGBoostClassifier(
        eval_metric = 'logloss',
        num_class = 3,
        nthread = 10,
        eta = 0.01,
        num_boost_round = 3000,
        max_depth = 9,
        subsample = 1.0,
        colsample_bytree = 0.9,
        silent = 0,
        )
    parameters = {
        'num_boost_round': [1000, 3000],
        'eta': [0.01, 0.05],
        'max_depth': [6, 9],
        'subsample': [0.9, 1.0],
        #'colsample_bytree': [0.9, 1.0],
    }
    
    if do_grid_search:
        clf = GridSearchCV(clf, parameters, n_jobs=5, cv=5, scoring='log_loss')
    '''
    -0.54009651324
    colsample_bytree: 0.9
    eta: 0.05
    max_depth: 6
    num_boost_round: 1000
    subsample: 1.0
    {'subsample': 1.0, 'num_boost_round': 1000, 'eta': 0.05, 'colsample_bytree': 0.9, 'max_depth': 6}

    '''
    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
        print(score)
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)        
    Y_pred = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)

    return Y_pred, Y_prob



def gbclassifier(training_samples, eval_samples, do_grid_search=True):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples

    X_train = X_train.toarray()
    X_eval = X_eval.toarray()
    clf = GradientBoostingClassifier(n_estimators=1000, loss='deviance',learning_rate=0.05, max_depth=9,
                                 min_weight_fraction_leaf=0.0, max_features=None, min_samples_split=4,
                                 min_samples_leaf=1, max_leaf_nodes=None,
                                 random_state=None, verbose=0, warm_start=False)
    to_be_tuned_parameters = {
                              #'n_estimators':[500, 1000, 2000, 4000],
                              'max_features':[50, 100, 150],
                              'min_samples_split':[10, 20, 40, 80],
                              #'min_samples_leaf': [1, 2],

                            }
    if do_grid_search:
        clf = GridSearchCV(clf, to_be_tuned_parameters, cv=5, n_jobs=5, scoring='log_loss')
    #Best parameters set found on development set:
    #()
    #{'max_features': None, 'min_samples_split': 10, 'n_estimators': 1000, 'min_samples_leaf': 2}
    
    
                

    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        print("Best parameters set found on development set:")
        print()
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=10, n_jobs=10, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)
    Y_eval = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)
    return Y_eval, Y_prob

def abclassifier(training_samples, eval_samples):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    do_grid_search=False
    clf = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=None, 
                                 min_samples_split=8, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                 max_features=40, max_leaf_nodes=None, bootstrap=True, oob_score=False, 
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)

    if do_grid_search:
        to_be_tuned_parameters = {
                                  'n_estimators':[500, 1000, 2000],
                                  'max_features':['log2', 'auto', None],
                                  'min_samples_split':[2, 4, 8],
                                  'min_samples_leaf': [1, 2],
    
                                  }
        clf = GridSearchCV(clf, to_be_tuned_parameters, cv=5, n_jobs=5, scoring='log_loss')

    #Best parameters set found on development set:
    #()
    #{'max_features': 'log2', 'min_samples_split': 8, 'criterion': 'gini', 'min_samples_leaf': 1}
    
                
    clf = AdaBoostClassifier(base_estimator=clf, n_estimators=200, learning_rate=0.2, algorithm='SAMME.R', random_state=None)
    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        print("Best parameters set found on development set:")
        print()
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)
    Y_eval = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)
    return Y_eval, Y_prob, clf.feature_importances_()

def knnclassifier(training_samples, eval_samples, do_grid_search=True):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    
    clf = KNeighborsClassifier(n_neighbors=120, leaf_size=1)

    if do_grid_search:
        to_be_tuned_parameters = {
                                  'n_neighbors':[30, 60, 90, 120, 180],
                                  'leaf_size':[1, 2, 5]
                                  }
        clf = GridSearchCV(clf, to_be_tuned_parameters, cv=5, n_jobs=5, scoring='log_loss')

    #Best parameters set found on development set:
    #()
    #{'max_features': 'log2', 'min_samples_split': 8, 'criterion': 'gini', 'min_samples_leaf': 1}
    
                
    print(clf)
    clf.fit(X_train, Y_train)
    if do_grid_search:
        print("Best parameters set found on development set:")
        print()
        
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
    else:
        scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
        print scores, np.mean(scores), np.median(scores)
    Y_eval = clf.predict(X_eval)
    Y_prob = clf.predict_proba(X_eval)
    return Y_eval, Y_prob

def blend_classifiers(training_samples, eval_samples):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    
    clfs = []
    clfs.append(
                KNeighborsClassifier(n_neighbors=120, leaf_size=1)
                )
    '''
    clfs.append(
                GradientBoostingClassifier(loss='deviance',learning_rate=0.05, max_depth=3, n_estimators=1000,
                                 min_weight_fraction_leaf=0.0, max_features=None, min_samples_split=4,
                                 min_samples_leaf=2, max_leaf_nodes=None,
                                 random_state=None, verbose=0, warm_start=False)
                
                )
    
    clfs.append(
                XGBoostClassifier(
                    eval_metric = 'logloss',
                    num_class = 3,
                    nthread = 10,
                    eta = 0.01,
                    num_boost_round = 3000,
                    max_depth = 9,
                    subsample = 0.9,
                    colsample_bytree = 0.9,
                    silent = 0,
                    )       
                )
    '''
    clfs.append(
                ExtraTreesClassifier(max_depth=None, n_estimators=2000,
                                 min_weight_fraction_leaf=0.0, max_features=None, min_samples_split=10, criterion='gini',
                                 min_samples_leaf=2, max_leaf_nodes=None, oob_score=False, bootstrap=True,
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)
                
                )
    clfs.append(
                RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=None, 
                                 min_samples_split=16, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                 max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, 
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)

                )
    clfs.append(
                ExtraTreesClassifier(max_depth=None, n_estimators=2000,
                                 min_weight_fraction_leaf=0.0, max_features=None, min_samples_split=10, criterion='entropy',
                                 min_samples_leaf=2, max_leaf_nodes=None, oob_score=False, bootstrap=True,
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)
                
                )
    clfs.append(
                RandomForestClassifier(n_estimators=2000, criterion='entropy', max_depth=None, 
                                 min_samples_split=16, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                 max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, 
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)
                
                )



    np.random.seed(0) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = False

    X, y, X_submission = X_train, Y_train, X_eval

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    X = np.array(X)
    X_submission = np.array(X_submission)
    y = np.array(y)
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i

            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    #y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    #print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    
    #print "Saving Results."
    #np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')
    
    Y_prob = clf.predict_proba(dataset_blend_test)
    Y_eval = clf.predict(dataset_blend_test)
    
    return Y_eval, Y_prob
    
def lassocvclassifier(training_samples, eval_samples, vectorizer, do_grid_search=False):
    X_train, Y_train = training_samples
    X_eval, Y_eval = eval_samples
    #clf = SGDClassifier(loss='log', penalty= 'l2',l1_ratio=0.0, n_iter=30, shuffle=True, verbose=False, 
    #                    n_jobs=4, alpha=1e-4, average=True, class_weight=None)
    clf = LassoCV()
   
    clf.fit(X_train, Y_train)
    #y_train_true, y_train_pred = Y_train, clf.predict(X_train)
    print_top_10_words = True
    
    
    scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=5, scoring='log_loss')
    print scores, np.mean(scores), np.median(scores)

    print(clf)
    #scores = cross_validation.cross_val_score(clf.best_estimator_, X_train, Y_train, cv=10, scoring='log_loss')
    #print scores, np.mean(scores), np.median(scores)
    y_true, y_pred = Y_eval, clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)