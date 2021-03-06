import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.neural_network import MLPClassifier
from time import perf_counter
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import accuracy_score
import mlrose_hiive
from sklearn.model_selection import cross_val_score



X_train_scaled[1]
algo = "NN"
algoName = "./census/census_"+ algo +"_"
data = './modified_data_orig'
datatype = "census"

bank_df  = pd.read_csv(data, delimiter=',')
bank_df = bank_df.drop(['native-country'], axis=1)
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
bank_df_dummies = pd.get_dummies(bank_df, columns=cat_vars)
bank_df_dummies['sallary'] = bank_df_dummies['sallary'].map({'>50K':0, '<=50K': 1})

labels = bank_df_dummies[['sallary']]
features = bank_df_dummies.drop(['sallary'], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, stratify=labels)

X_train_scaled, X_test_scaled, y_train_hot, y_test_hot = train_test_split(features, labels, test_size=0.20, stratify=labels)

clf = MLPClassifier(hidden_layer_sizes=(9,9), activation = 'relu', max_iter=1000, solver = 'lbfgs' )
time_start = perf_counter()
clf.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train: fit_time = {fit_time}')

train_accuracy = accuracy_score(y_train, )
print (f'train accuracy score = {train_accuracy}')


time_start = perf_counter()
yhat = clf.predict(X_test_scaled)
fit_time = perf_counter() - time_start
print (f'Test: fit_time = {fit_time}')

#Evaulation
jaccard = jaccard_similarity_score(y_test_hot, yhat)
print("jaccard index: ",jaccard)
print (classification_report(y_test_hot, yhat))





np.random.seed(7)
clf_hill = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_hill.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_hill_test = clf_hill.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy1 = accuracy_score(y_test_hot, yhat_hill_test)
f1 = f1_score(y_test_hot, yhat_hill_test, average='weighted') 
jaccard1 = jaccard_similarity_score(y_test_hot, yhat_hill_test)
print (f'fit_time = {fit_time}')
print (f'accuracy score = {test_accuracy1}')
print("f1 score: ", f1)
print("jaccard index: ",jaccard1)
print (classification_report(y_test_hot, yhat_hill_test))


np.random.seed(7)
clf_hill_sig = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_hill_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')

#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_hill_test_sig = clf_hill_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy2 = accuracy_score(y_test_hot, yhat_hill_test_sig)
f2 = f1_score(y_test_hot, yhat_hill_test_sig, average='weighted') 
jaccard2 = jaccard_similarity_score(y_test_hot, yhat_hill_test_sig)
print (f'fit_time = {fit_time}')
print (f'accuracy score = {test_accuracy2}')
print("f1 score: ", f2)
print("jaccard index: ",jaccard2)
print (classification_report(y_test_hot, yhat_hill_test_sig))

np.random.seed(7)
clf_sim = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')



#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test = clf_sim.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy3 = accuracy_score(y_test_hot, yhat_sim_test)
f3 = f1_score(y_test_hot, yhat_sim_test, average='weighted') 
jaccard3 = jaccard_similarity_score(y_test_hot, yhat_sim_test)
print (f'fit_time = {fit_time}')
print (f'test accuracy score = {test_accuracy3}')
print("f1 score: ", f3)
print("jaccard index: ",jaccard3)
print (classification_report(y_test_hot, yhat_sim_test))

#np.random.seed(7)
clf_sim_sig = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')

#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test_sig = clf_sim_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy4 = accuracy_score(y_test_hot, yhat_sim_test_sig)
f4 = f1_score(y_test_hot, yhat_sim_test_sig, average='weighted') 
jaccard4 = jaccard_similarity_score(y_test_hot, yhat_sim_test_sig)
print (f'fit_time = {fit_time}')
print (f'test accuracy score = {test_accuracy4}')
print("f1 score: ", f4)
print("jaccard index: ",jaccard4)
print (classification_report(y_test_hot, yhat_sim_test_sig))


#np.random.seed(7)
clf_gen = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'genetic_alg', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_gen.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')

#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_gen_test = clf_gen.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy5 = accuracy_score(y_test_hot, yhat_gen_test)
f5 = f1_score(y_test_hot, yhat_gen_test, average='weighted') 
jaccard5 = jaccard_similarity_score(y_test_hot, yhat_gen_test)
print (f'fit_time = {fit_time}')
print (f'accuracy score = {test_accuracy5}')
print("f1 score: ", f5)
print("jaccard index: ",jaccard5)

print (classification_report(y_test_hot, yhat_gen_test))

#np.random.seed(7)
clf_gen_sig = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'genetic_alg', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_gen_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'fit_time = {fit_time}')


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_gen_test_sig = clf_gen_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy6 = accuracy_score(y_test_hot, yhat_gen_test_sig)
f6 = f1_score(y_test_hot, yhat_gen_test_sig, average='weighted') 
jaccard6 = jaccard_similarity_score(y_test_hot, yhat_gen_test_sig)
print (f'fit_time = {fit_time}')
print (f'accuracy score = {test_accuracy6}')
print("f1 score: ", f6)
print("jaccard index: ",jaccard6)

print (classification_report(y_test_hot, yhat_gen_test_sig))