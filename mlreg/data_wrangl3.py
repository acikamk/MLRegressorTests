import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy import sparse 
from sklearn.datasets import make_classification, make_blobs, load_boston, fetch_california_housing 
from sklearn.decomposition import PCA
from sklearn import metrics 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV 
from pprint import pprint 
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from nltk import PorterStemmer
from nltk.stem.snowball import GermanStemmer
from sklearn import preprocessing
from sklearn.svm import SVR
from numpy import loadtxt
from xgboost import XGBRegressor	
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.kernel_ridge import KernelRidge
from scipy.sparse import hstack


data = pd.read_csv('~/Downloads/ml-price-prediction-dataset.csv', sep=';', low_memory=False, encoding='latin')
test_df = data[['restaurant_name','city_id','postcode','menu_category','product_name', 'price']].copy()
test_df.dropna(inplace=True)
test_df.reset_index(inplace=True)
print test_df.isnull().sum()


encode_menu = test_df['menu_category'].str.encode('ascii', errors='ignore')
print len(encode_menu.unique())
encode_menu.replace({r'[^a-zA-Z0-9\s,]':''}, regex=True, inplace=True)
print len(encode_menu.unique())
encode_menu = encode_menu.apply(lambda x:GermanStemmer().stem(x))
print len(encode_menu.unique())
encode_name = test_df['product_name'].str.encode('ascii', errors='ignore')
print len(encode_name.unique())
encode_name.replace({r'[^a-zA-Z0-9\s,]':''}, regex=True, inplace=True)
print len(encode_name.unique())
encode_name = encode_name.apply(lambda x:GermanStemmer().stem(x))
print len(encode_name.unique())

encode_rname = test_df['restaurant_name'].str.encode('ascii', errors='ignore')
encode_rname.replace({r'[^a-zA-Z0-9\s,]':''}, regex=True, inplace=True)
print len(encode_rname.unique())
encode_rname = encode_rname.apply(lambda x:GermanStemmer().stem(x))
print len(encode_rname.unique())
# test_df['postcode']=test_df['postcode'].apply(lambda x: x.split()[0])
test_df['postcode'].replace({r'[^0-9]':''}, regex=True, inplace=True)
# tfidf_vectorizer = TfidfVectorizer(max_df=18000, min_df=20)
# t_name = tfidf_vectorizer.fit_transform(encode_menu)
# tfidf_vectorizer = TfidfVectorizer(max_df=2000, min_df=20)
# t_menu = tfidf_vectorizer.fit_transform(encode_menu)
# X =	pd.concat((pd.DataFrame(t_name.toarray()), pd.DataFrame(t_menu.toarray()), test_df[['restaurant_id','city_id','postcode']]), axis=1)

seed = 7
test_size = 0.2

X = pd.concat((encode_rname, encode_menu, encode_name, test_df['city_id'], test_df['postcode']), axis=1)
y = test_df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# print X.shape
# print X_train.shape
# pdb.set_trace()
# X_train.columns = ['rname','menu_cat','pname','city_id','postcode']
# X_test.columns = ['rname','menu_cat','pname','city_id','postcode']

# pdb.set_trace()		

# word_vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='char',strip_accents='ascii', token_pattern='\w|,|\s|')
# smm = word_vectorizer.fit_transform(encode_name)
# smn= word_vectorizer.fit_transform(encode_menu)
# smr= word_vectorizer.fit_transform(encode_rname)
# X =	pd.concat((pd.DataFrame(smr.toarray()), pd.DataFrame(smm.toarray()), pd.DataFrame(smn.toarray())), axis=1)
# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# X_norm = scale.fit_transform(X)
# print X_norm.shape
# pdb.set_trace()

# pca = PCA(copy=True, iterated_power='auto', random_state=None, svd_solver='auto', tol=0.0, whiten=False)
# pca.fit(X_norm)
# out = pca.fit_transform(X_norm)
# pca_comp=pca.components_
# plt.figure()
# plt.plot(pca_comp[0], 'b')
# plt.plot(pca_comp[1], 'r')
# # sc = plt.scatter(out[:,0], out[:,1], c=y, vmin=0, vmax=20)
# # plt.colorbar(sc)
# plt.show()
# pdb.set_trace()
# plt.figure()
# X_tnse=TSNE(n_components=2, verbose=2)
# X_tnse.fit_transform(X_norm)	
# sc = plt.scatter(X_tnse[:,0], X_tnse[:,1], c=y, vmin=0, vmax=20)
# plt.colorbar(sc)
# seed = 7
# test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=test_size, random_state=seed)
# # fit model no training data
# model = XGBRegressor()
# model.fit(X_train, y_train)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
 
# split train teast before text manupulation
# word_vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='char',strip_accents='ascii', token_pattern='\w|,|\s|')
word_vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='char',strip_accents='ascii', token_pattern='\w|,|\s|')
# smm = word_vectorizer.fit_transform(encode_name)
# smn= word_vectorizer.fit_transform(encode_menu)
# smr= word_vectorizer.fit_transform(encode_rname)
# pdb.set_trace()
smr = word_vectorizer.fit_transform(X_train['restaurant_name'])
smm= word_vectorizer.fit_transform(X_train['menu_category'])
smp= word_vectorizer.fit_transform(X_train['product_name'])
pdb.set_trace()
rdf=pd.DataFrame(smr.toarray()) 
rdf.columns= ['r_'+str(el) for el in rdf.columns]
mdf=pd.DataFrame(smm.toarray()) 
mdf.columns= ['m_'+str(el) for el in mdf.columns]
pdf=pd.DataFrame(smp.toarray()) 
pdf.columns= ['p_'+str(el) for el in pdf.columns]

city_id = pd.get_dummies(X_train['city_id'])
city_id.columns = ['city_id_'+str(el) for el in city_id.columns]
city_id.reset_index(inplace=True)
city_id.drop(labels=['index'], axis=1,inplace=True)
cid_csc = city_id.values.tocsr()

postcode =pd.get_dummies(X_train['postcode'])
postcode.columns = ['post_'+str(el) for el in postcode.columns]
postcode.reset_index(inplace=True)
postcode.drop(labels=['index'], axis=1, inplace=True)
p_scs = postcode.values.tocsr()

pdb.set_trace()
# X_train=pd.concat((rdf,mdf,pdf,city_id,postcode), axis=1) 
X_train=hstack([	rdf,mdf,pdf,cid_csc,p_scs])


# X_train=pd.concat((pd.DataFrame(smr.toarray()), pd.DataFrame(smm.toarray()), pd.DataFrame(smn.toarray()), X_train['city_id'], X_train['postcode']), axis=1) 
print X_train.shape
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_norm = scale.fit_transform(X_train)
print X_norm.shape



# Test
smrt= word_vectorizer.fit_transform(X_test['restaurant_name'])
smmt= word_vectorizer.fit_transform(X_test['menu_category'])
smpt= word_vectorizer.fit_transform(X_test['product_name'])
rdf=pd.DataFrame(smrt.toarray()) 
rdf.columns= ['r_'+str(el) for el in rdf.columns]
mdf=pd.DataFrame(smmt.toarray()) 
mdf.columns= ['m_'+str(el) for el in mdf.columns]
pdf=pd.DataFrame(smpt.toarray()) 
pdf.columns= ['p_'+str(el) for el in pdf.columns]

city_id = pd.get_dummies(X_test['city_id'])

city_id.columns = ['city_id_'+str(el) for el in city_id.columns]

city_id.reset_index(inplace=True)
city_id.drop(labels=['index'], axis=1,inplace=True)

postcode =pd.get_dummies(X_test['postcode'])
postcode.columns = ['post_'+str(el) for el in postcode.columns]
postcode.reset_index(inplace=True)
postcode.drop(labels=['index'], axis=1,inplace=True)

print 'postcode {}'.format(postcode.shape)
X_test=pd.concat((rdf,mdf,pdf,city_id, postcode), axis=1) 

print 'test {}'.format(X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_normt = scale.fit_transform(X_test)
print X_normt.shape


# model = XGBRegressor()
# model.fit(X_train, y_train)
xgb_cv = GridSearchCV(XGBRegressor(), cv=3, param_grid={'max_depth':[3,10], 'learning_rate':[0.1, 1], 'n_estimators':[50,100,150]})
xgb_cv.fit(X_norm, y_train)
pdb.set_trace()
best=xgb_cv.best_estimator_

xgb_train_score=xgb_cv.score(X_train, y_train)
xgb_test_score=xgb_cv.score(X_test, y_test)
print xgb_train_score
print xgb_test_score
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=10, min_child_weight=1, missing=None, n_estimators=150,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
# best.score(X_train, y_train)
# 0.91050046265714446

# 0.916120904111
# 0.485837303997


# xgb_cvn = GridSearchCV(XGBRegressor(), cv=3, param_grid={'max_depth':[3,10], 'learning_rate':[0.1, 1], 'n_estimators':[50,100,150]})
# xgb_cvn.fit(X_norm, y_train)
# bestn=xgb_cvn.best_best_estimator_
# pdb.set_trace()
# xgbn_train_score=xgb_cv.score(X_norm, y_train)
# xgbn_test_score=xgb_cv.score(X_normt, y_test)
# print xgbn_train_score
# print xgbn_test_score


# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
# model.score(X_train, y_train)   
# 0.46557036599497936
# model.score(X_test, y_test) 


# (Pdb) xgb_cv.best_estimator_
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=10, min_child_weight=1, missing=None, n_estimators=150,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
# (Pdb) xgb_cv.best_params_
# {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 10}
# (Pdb) xgb_cv.cv_results_['split0_test_score']
# array([ 0.26322395,  0.31619848,  0.33569375,  0.41949536,  0.44468269,
#         0.45294125,  0.25672058,  0.27799842,  0.26278467,  0.25142055,
#         0.24984566,  0.24995027])
# (Pdb) xgb_cv.cv_results_['split1_test_score']
# array([ 0.28329181,  0.30788279,  0.31820614,  0.33439889,  0.33676339,
#         0.33842815,  0.23780545,  0.21851832,  0.22662203,  0.2175756 ,
#         0.21500993,  0.21388094])
# (Pdb) xgb_cv.cv_results_['split1_train_score']
# array([ 0.41220895,  0.50148945,  0.54342967,  0.87322433,  0.9085871 ,
#         0.92843675,  0.65738876,  0.74749185,  0.7874091 ,  0.97724889,
#         0.994991  ,  0.99876116])
# (Pdb) xgb_cv.cv_results_['split0_train_score']
# array([ 0.39233953,  0.46836535,  0.50222442,  0.87062457,  0.90277793,
#         0.92143319,  0.62988185,  0.69989503,  0.75245609,  0.97667526,
#         0.99182389,  0.99514325])


# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)


# (Pdb) rf.cv_results_['split1_train_score']
# array([ 0.31504952,  0.45117987,  0.67897048])
# (Pdb) rf.cv_results_['split1_test_score']
# array([ 0.17369633,  0.24552573,  0.31034079])
# (Pdb) rf.cv_results_['split0_test_score']
# array([ 0.20841938,  0.23836207,  0.27071412])
# (Pdb) rf.cv_results_['split0_train_score']
# array([ 0.30856587,  0.47459398,  0.6767325 ])

# (Pdb) from sklearn.preprocessing import MinMaxScaler
# (Pdb) scale = MinMaxScaler()
# (Pdb) X_norm = scale.fit_transform(X_train)
# (Pdb) rf=GridSearchCV(RandomForestRegressor(), cv=3, param_grid={'n_estimators':[100,150],'max_depth':[4,6,10]})
# (Pdb) rf.fit(X_norm, y_train) 
# GridSearchCV(cv=3, error_score='raise',
#        estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False),
#        fit_params=None, iid=True, n_jobs=1,
#        param_grid={'n_estimators': [100, 150], 'max_depth': [4, 6, 10]},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
#        scoring=None, verbose=0)
# (Pdb) rf.score(X_norm, y_train)
# 0.63648816563703936
# (Pdb) rf.score(X_rest, y_test)
# *** NameError: name 'X_rest' is not defined
# (Pdb) rf.score(X_test, y_test)
# *** ValueError: could not convert string to float: 
# (Pdb) rf.best_estimator
# *** AttributeError: 'GridSearchCV' object has no attribute 'best_estimator'
# (Pdb) rf.best_estimator_
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)
# (Pdb) rf.cv_results_
# {'std_train_score': array([ 0.04376343,  0.04616774,  0.02833745,  0.0227273 ,  0.01377272,
#         0.01024641]), 'rank_test_score': array([5, 6, 4, 3, 1, 2], dtype=int32), 'param_max_depth': masked_array(data = [4 4 6 6 10 10],
#              mask = [False False False False False False],
#        fill_value = ?)
# , 'split1_train_score': array([ 0.35605474,  0.3518186 ,  0.5057006 ,  0.51974606,  0.62651116,
#         0.6340184 ]), 'split2_train_score': array([ 0.41374335,  0.41586705,  0.51818522,  0.52014248,  0.64993207,
#         0.65225832]), 'std_score_time': array([ 0.01332746,  0.02144588,  0.04170102,  0.04115009,  0.01485711,
#         0.05242565]), 'split2_test_score': array([ 0.30581357,  0.31776694,  0.37004168,  0.36718886,  0.43910952,
#         0.44849732]), 'mean_score_time': array([ 0.46236205,  0.62592228,  0.63310099,  0.93072454,  1.08910473,
#         1.6276234 ]), 'mean_fit_time': array([ 185.52156798,  242.76824832,  197.79265364,  301.24429401,
#         325.27815127,  485.88149953]), 'split0_train_score': array([ 0.46314602,  0.46455785,  0.57107539,  0.56815493,  0.65924996,
#         0.65806917]), 'std_test_score': array([ 0.06292765,  0.05601493,  0.06196299,  0.06208554,  0.06326572,
#         0.07232768]), 'mean_train_score': array([ 0.41098137,  0.41074783,  0.53165374,  0.53601449,  0.64523107,
#         0.6481153 ]), 'split0_test_score': array([ 0.18988982,  0.19022642,  0.2531768 ,  0.25589716,  0.3333159 ,
#         0.31725211]), 'mean_test_score': array([ 0.2771789 ,  0.26860105,  0.33956616,  0.34146213,  0.41890193,
#         0.41722882]), 'params': [{'n_estimators': 100, 'max_depth': 4}, {'n_estimators': 150, 'max_depth': 4}, {'n_estimators': 100, 'max_depth': 6}, {'n_estimators': 150, 'max_depth': 6}, {'n_estimators': 100, 'max_depth': 10}, {'n_estimators': 150, 'max_depth': 10}], 'std_fit_time': array([ 12.32584503,  16.64777944,  13.94742488,  17.26114665,
#          2.74066178,  21.78782425]), 'param_n_estimators': masked_array(data = [100 150 100 150 100 150],
#              mask = [False False False False False False],
#        fill_value = ?)
# , 'split1_test_score': array([ 0.33583332,  0.29780978,  0.39548   ,  0.40130037,  0.48428037,
#         0.48593704])}
