'''
Created on 7 Jan 2016

@author: af
'''
import csv
from os import path
import pdb
from _collections import defaultdict
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import models
from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
import itertools
import sklearn.preprocessing as prep 
from sklearn.decomposition import PCA
from sklearn.decomposition.nmf import NMF
from sklearn.feature_selection import RFECV
from sklearn.ensemble.forest import RandomForestClassifier

severity_type_file = 'severity_type.csv'
train_file = 'train.csv'
test_file = 'test.csv'
log_feature_file = 'log_feature.csv'
resource_type_file = 'resource_type.csv'
event_type_file = 'event_type.csv'
data_home = '/home/arahimi/datasets/telstra'
files = [severity_type_file, train_file, test_file, log_feature_file, resource_type_file, event_type_file]


def readfile(filename):
    with open(filename, 'r') as inf:
        lines = inf.readlines()
        #ignore the first line (column titles)
        lines = lines[1:]
    return lines

data = {f:readfile(path.join(data_home, f)) for f in files}
id_location_train = {l.strip().split(',')[0]:l.strip().split(',')[1] for l in data[train_file]}
id_severity_train = {l.strip().split(',')[0]:l.strip().split(',')[2] for l in data[train_file]}

id_location_test = {l.strip().split(',')[0]:l.strip().split(',')[1] for l in data[test_file]}

id_location = {x:y for x, y in id_location_train.iteritems()}
id_location.update(id_location_test)


id_event_type = defaultdict(list)
id_resource_type = defaultdict(list)
id_resource_count = defaultdict(int)
id_event_count = defaultdict(int)
id_severity_type = {}

    
for l in data[resource_type_file]:
    fields = l.strip().split(',')
    id = fields[0]
    resource_type = fields[1]
    id_resource_type[id].append(resource_type)

for l in data[severity_type_file]:
    fields = l.strip().split(',')
    id = fields[0]
    severity_type = fields[1]
    id_severity_type[id] = severity_type
    
id_event_type = defaultdict(list)
for l in data[event_type_file]:
    fields = l.strip().split(',')
    id = fields[0]
    event_type = fields[1]
    id_event_type[id].append(event_type)
id_event_count = {id:len(events) for id, events in id_event_type.iteritems()}
id_resource_count = {id:len(resources) for id, resources in id_resource_type.iteritems()}

id_features = defaultdict(dict)
feature_sum_train = defaultdict(int)
feature_sum_test = defaultdict(int)
total_feature_count = 0
add_combinations = True
add_location = False
for l in data[log_feature_file]:
    total_feature_count += 1
    fields = l.strip().split(',')
    id = fields[0]
    if id in id_location_train:
        feature_sum_train[fields[1]] += np.abs(int(fields[2]))
    else:
        feature_sum_test[fields[1]] += np.abs(int(fields[2]))

################################################################## label distribution of location of event ###############
#distribution of labels in each location of training data
add_label_distribution_of_location = False
if add_label_distribution_of_location:
    location_count = defaultdict(int)
    locationlabel_count = defaultdict(int)
    for l in data[train_file]:
        id, location, label = l.split(',')
        location_count[location] += 1
        locationlabel_count[location + label] += 1
    for l in data[train_file]:
        id, location, label = l.split(',')
        id_features[id]['loclab0'] = locationlabel_count[location+'0'] #/ float(location_count[location])
        id_features[id]['loclab1'] = locationlabel_count[location+'1'] #/ float(location_count[location])
        id_features[id]['loclab2'] = locationlabel_count[location+'2'] #/ float(location_count[location])
        #id_features[id]['location_count'] = location_count[location]
#count the number of features for each event and use it as a feature

normalise_features = False
id_feature_count = defaultdict(int)
for l in data[log_feature_file]:
    fields = l.strip().split(',')
    id = fields[0]
    if normalise_features:
        if id in id_location_train:
            sum_feature = feature_sum_train[fields[1]]
        else:
            sum_feature = feature_sum_test[fields[1]]
    else:
        sum_feature = 1.0

    id_features[fields[0]][fields[1]] = float(fields[2]) / float(sum_feature)
    id_feature_count[fields[0]] += 1

all_ids = sorted(id_location.keys())
for id in all_ids:
    
    events_dic = {e:'1' for e in id_event_type[id]}
    id_features[id].update(events_dic)

    resources_dic = {r:'1' for r in id_resource_type[id]}
    id_features[id].update(resources_dic)
    
    id_features[id][id_severity_type[id]] = '1'
    if add_combinations:
        combination_features = {}
        '''
        for i in range(2, len(resources_dic) + 1):
            combs_i = itertools.combinations(resources_dic.keys(), i)
            combs_i_dict = {str(f):'1' for f in combs_i}
            combination_features.update(combs_i_dict)

        
        for i in range(2, len(events_dic) + 1):
            combs_i = itertools.combinations(events_dic.keys(), i)
            combs_i_dict = {str(f):'1' for f in combs_i}
            combination_features.update(combs_i_dict)
        
        '''
        #all_categorical_features = events_dic.keys() + resources_dic.keys() + [id_severity_type[id]] + [id_location[id]]
        all_categorical_features = events_dic.keys() + resources_dic.keys() + [id_severity_type[id]]
        maximum_feature_length = len(all_categorical_features)
        maximum_feature_length = 3
        for i in range(2, maximum_feature_length):
            combs_i = itertools.combinations(all_categorical_features, i)
            combs_i_dict = {str(f):'1' for f in combs_i}
            combination_features.update(combs_i_dict)
        id_features[id].update(combination_features)    
    if add_location:                  
        id_features[id][id_location[id]] = '1' 
    id_features[id]['feature_count'] = float(id_feature_count[id])
    id_features[id]['event_count'] = id_event_count[id]
    id_features[id]['resource_count'] = id_resource_count[id]
    
    
train_ids = sorted(id_location_train.keys())
test_ids = sorted(id_location_test.keys())
train_features = [id_features[id] for id in train_ids]
test_features = [id_features[id] for id in test_ids]
labels = {'0':0, '1':1, '2':2}
train_labels = [labels[id_severity_train[id]] for id in train_ids]
test_fake_labels = [train_labels[0]] * len(test_ids)
vectorizer = DictVectorizer()

X_train = vectorizer.fit_transform(train_features)
features = vectorizer.get_feature_names()
save_train_features = False
if save_train_features:
    np.savetxt('x_train.txt', X_train.toarray(), delimiter=',', header=','.join(features))

X_test = vectorizer.transform(test_features)

#scaler = prep.MinMaxScaler(feature_range=(0, 1), copy=True)
scaler = prep.StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train.toarray())
X_test = scaler.transform(X_test.toarray())

do_feature_elimination = False
if do_feature_elimination:
    estimator =  RandomForestClassifier(n_estimators=2000, criterion='entropy', max_depth=None, 
                                 min_samples_split=16, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                 max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, 
                                 n_jobs=10, random_state=None, verbose=0, warm_start=False, class_weight=None)
    selector = RFECV(estimator, step=1, cv=5, scoring='log_loss')
    X_train = selector.fit_transform(X_train, train_labels)
    print 'after feature elimination', X_train.shape
    X_test = selector.transform(X_test)
    
do_feature_selection = False
if do_feature_selection:
    ch2 = SelectKBest(chi2, k=4000)
    X_train = ch2.fit_transform(X_train, train_labels)
    X_test = ch2.transform(X_test)

do_pca = False

if do_pca:
    k = 100
    add_pca_to_original = True
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    pca = PCA(n_components=k, copy=True, whiten=False)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    if add_pca_to_original:
        X_train = np.hstack((X_train, X_train_pca))
        X_test = np.hstack((X_test, X_test_pca))
    else:
        X_train = X_train_pca
        X_test = X_test_pca

#X_train = normalize(X_train, norm='l1', axis=1)

#X_train = normalize(X_train, norm='l1', axis=0)

#X_train = normalise(X_train)
#X_test = normalise(X_test)

#X_test = normalize(X_test, norm='l1', axis=1)
#X_test = normalize(X_test, norm='l1', axis=0)
#k = int(X_train.shape[0] / 10)
#X_test = X_train[0:k]
#X_train = X_train[k:]
#train_labels = train_labels[k:]
#test_fake_labels=train_labels[0:k]
combine_all = False
if not combine_all:
    print 'x_train shape is: ', X_train.shape
    model_names = ['lr', 'rf', 'xg', 'et', 'gb', 'ab', 'knn', 'blend']
    model_name = 'xg'
    if model_name == 'lr':
        Y_eval, Y_prob = models.sgdclassifier((X_train, train_labels), (X_test, test_fake_labels), vectorizer)
        result_file = 'results-lr.txt'
    elif model_name == 'rf':
        result_file = 'results-rf.txt'
        Y_eval, Y_prob, feature_importance = models.rfclassifier((X_train, train_labels), (X_test, test_fake_labels))
        indices = np.argsort(feature_importance)
        feats = np.array(features)[indices].tolist()
        feats.reverse()
        print feats[0: 50]
    
    elif model_name == 'xg':
        result_file = 'results-xg.txt'
        Y_eval, Y_prob = models.xgboostclassifier((X_train, train_labels), (X_test, test_fake_labels))
    elif model_name == 'et':
        result_file = 'results-et.txt'
        Y_eval, Y_prob = models.etclassifier((X_train, train_labels), (X_test, test_fake_labels))
    
    elif model_name == 'gb':
        result_file = 'results-gb.txt'
        Y_eval, Y_prob = models.gbclassifier((X_train, train_labels), (X_test, test_fake_labels))
    
    elif model_name == 'ab':
        result_file = 'results-ab.txt'
        Y_eval, Y_prob, feature_importance = models.abclassifier((X_train, train_labels), (X_test, test_fake_labels))
        indices = np.argsort(feature_importance)
        feats = np.array(features)[indices].tolist()
        feats.reverse()
    elif model_name == 'knn':
        result_file = 'results-knn.txt'
        Y_eval, Y_prob = models.knnclassifier((X_train, train_labels), (X_test, test_fake_labels))
    elif model_name == 'blend':
        result_file = 'results-blend.txt'
        Y_eval, Y_prob = models.blend_classifiers((X_train, train_labels), (X_test, test_fake_labels))
    if model_name in model_names:
        with open('scores_' + model_name + '.pkl', 'wb') as outf:
            pickle.dump((Y_eval, Y_prob), outf)



if combine_all:
    Y_all_probs = []
    for name in ['rf', 'xg', 'gb']:
        with open('scores_' + name + '.pkl', 'rb') as inf:
            Y_pred, Y_prob = pickle.load(inf)
            Y_prob = np.nan_to_num(Y_prob)
            Y_all_probs.append(Y_prob)
    Y_prob = Y_all_probs[0]
    for i in range(1, len(Y_all_probs)):
        Y_prob += Y_all_probs[i]
    Y_prob = Y_prob / len(Y_all_probs)
    result_file = 'results_all.txt'
binary = False
with open(path.join(data_home, result_file), 'w') as outf:
    outf.write('id,predict_0,predict_1,predict_2\n')
    for i, id in enumerate(test_ids):
        probs = Y_prob[i]
        first_prob = probs[0]
        second_prob = probs[1]
        third_prob = probs[2]
        if binary:
            all_values = [first_prob, second_prob, third_prob]
            max_prob = max(all_values)
            if first_prob < max_prob:
                first_prob = 0
            else:
                first_prob = 1
            if second_prob < max_prob:
                second_prob = 0
            else:
                second_prob = 1
            if third_prob < max_prob:
                third_prob = 0
            else:
                third_prob = 0
        outf.write(id + ',' + str(first_prob) + ',' + str(second_prob) + ',' + str(third_prob) + '\n')
