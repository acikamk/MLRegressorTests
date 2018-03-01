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
test_df.dropna(subset=['restaurant_name','menu_category','product_name','postcode'], inplace=True)
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

# tfidf_vectorizer = TfidfVectorizer(max_df=18000, min_df=20)
# t_name = tfidf_vectorizer.fit_transform(encode_menu)
# tfidf_vectorizer = TfidfVectorizer(max_df=2000, min_df=20)
# t_menu = tfidf_vectorizer.fit_transform(encode_menu)
# X =	pd.concat((pd.DataFrame(t_name.toarray()), pd.DataFrame(t_menu.toarray()), test_df[['restaurant_id','city_id','postcode']]), axis=1)

y = test_df['price'].values

word_vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='char',strip_accents='ascii', token_pattern='\w|,|\s|')
smm = word_vectorizer.fit_transform(encode_name)
smn= word_vectorizer.fit_transform(encode_menu)
smr= word_vectorizer.fit_transform(encode_rname)
X =	pd.concat((pd.DataFrame(smr.toarray()), pd.DataFrame(smm.toarray()), pd.DataFrame(smn.toarray())), axis=1)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_norm = scale.fit_transform(X)
print X_norm.shape


pca = PCA(copy=True, iterated_power='auto', random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca.fit(X_norm)
out = pca.fit_transform(X_norm)
pca_comp=pca.components_
plt.figure()
plt.plot(pca_comp[0], 'b')
plt.plot(pca_comp[1], 'r')
# sc = plt.scatter(out[:,0], out[:,1], c=y, vmin=0, vmax=20)
# plt.colorbar(sc)
plt.show()
pdb.set_trace()
plt.figure()
X_tnse=TSNE(n_components=2, verbose=2)
X_tnse.fit_transform(X_norm)	
sc = plt.scatter(X_tnse[:,0], X_tnse[:,1], c=y, vmin=0, vmax=20)
plt.colorbar(sc)
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



