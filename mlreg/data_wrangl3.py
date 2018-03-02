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
word_vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='char',strip_accents='ascii', token_pattern='\w|,|\s|')
# smm = word_vectorizer.fit_transform(encode_name)
# smn= word_vectorizer.fit_transform(encode_menu)
# smr= word_vectorizer.fit_transform(encode_rname)
# pdb.set_trace()
smrt = word_vectorizer.fit_transform(X_train['restaurant_name'])
smmt= word_vectorizer.fit_transform(X_train['menu_category'])
smpt= word_vectorizer.fit_transform(X_train['product_name'])
rdf=pd.DataFrame(smrt.toarray()) 
rdf.columns= ['r_'+str(el) for el in rdf.columns]
mdf=pd.DataFrame(smmt.toarray()) 
mdf.columns= ['m_'+str(el) for el in mdf.columns]
pdf=pd.DataFrame(smpt.toarray()) 
pdf.columns= ['p_'+str(el) for el in pdf.columns]

city_id = pd.get_dummies(X_train['city_id'])
city_id.columns = ['city_id_'+str(el) for el in city_id.columns]
city_id.reset_index(inplace=True)
city_id.drop(labels=['index'], axis=1,inplace=True)
postcode =pd.get_dummies(X_train['postcode'])

postcode.columns = ['post_'+str(el) for el in postcode.columns]
postcode.reset_index(inplace=True)
postcode.drop(labels=['index'], axis=1, inplace=True)

pdb.set_trace()
X_train=pd.concat((rdf,mdf,pdf,city_id,postcode), axis=1) 


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


# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
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


