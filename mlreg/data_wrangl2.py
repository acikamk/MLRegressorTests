	import pandas as pd
import numpy as np
import matplotlib
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

from sklearn.kernel_ridge import KernelRidge
# import pandas as pd from pandas.tools.plotting 
# import scatter_matrix 
# import seaborn
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()		

data = pd.read_csv('~/Downloads/ml-price-prediction-dataset.csv', sep=';', low_memory=False, encoding='latin')
n_features = 1000
n_components = 1	
'''
Index([u'restaurant_id', u'restaurant_name', u'testing',
       u'restaurant_description', u'product_id', u'variation_id',
       u'menu_category', u'product_name', u'product_description', u'price',
       u'postcode', u'latitude', u'longitude', u'chain_id',
       u'restaurant_active', u'city_id', u'dietary_characteristics',
       u'cuisine_characteristics', u'taste_characteristics',
       u'preparation_style_characteristics', u'dish_type_characteristics',
       u'ingredients'],
      dtype='object')
'''
test_df = data[['restaurant_name','menu_category','product_name', 'price']].copy()
test_df.dropna(subset=['menu_category','product_name'], inplace=True)
test_df.reset_index(inplace=True)
print test_df.isnull().sum()



print 'Unique restaurants: {}'.format(len(data['restaurant_name'].unique()))
print 'Unique menu_category: {}'.format(len(data['menu_category'].unique()))
print 'Unique product_name: {}'.format(len(data['product_name'].unique()))
print 'Unique ingredients: {}'.format(len(data['ingredients'].unique()))
print test_df.shape

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


# X = pd.concat([encode_menu, encode_name, test_df['restaurant_id'].astype('str')], axis=1)

# le = preprocessing.LabelEncoder()
# X_2 = X.apply(le.fit_transform)
# print X_2.head()
# print X_2.shape
# enc = preprocessing.OneHotEncoder()

# # 2. FIT
# enc.fit(X_2)

# pdb.set_trace()	

# 3. Transform
# onehotlabels = enc.transform(X_2).toarray()
# print onehotlabels.shape


pdb.set_trace()
tfidf_vectorizer = TfidfVectorizer()
# X_train_v=tfidf_vectorizer.fit_transform(np.hstack(X_train.values))
# X_test_v=tfidf_vectorizer.fit_transform(np.hstack(X_test.values))
tfidf_vectorizer = TfidfVectorizer(max_df=18000, min_df=20)
t_name = tfidf_vectorizer.fit_transform(encode_menu)
tfidf_vectorizer = TfidfVectorizer(max_df=2000, min_df=20)
t_menu = tfidf_vectorizer.fit_transform(encode_menu)

tfidf_vectorizer = TfidfVectorizer()
t_rname = tfidf_vectorizer.fit_transform(encode_rname)
# X = t_name
# y = test_df[['price']].values


f.open('Scores_wrang2.txt', 'w')

 # new_features_name=pd.DataFrame(t_name.toarray())
# new_features_menu=pd.DataFrame(t_menu.toarray())

X =	pd.concat((pd.DataFrame(t_rname.toarray()), pd.DataFrame(t_name.toarray()), pd.DataFrame(t_menu.toarray()), test_df['restaurant_id']), axis=1)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_norm = scale.fit_transform(X)

y = test_df['price'].values

pdb.set_trace()	
X_train, X_test, y_train, y_test = train_test_split(X, y)

cv = KFold(3)
ridge = RidgeCV(cv=cv)

ridge.fit(X_train, y_train)

R_ridge_train_score = ridge.score(X_train, y_train)
R_ridge_test_score = ridge.score(X_test, y_test)
print R_ridge_train_score, R_ridge_test_score

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3,
                   param_grid={"C": [1e0, 1e1, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

svr.fit(X_train, y_train)
svr_train_score=svr.score(X_train, y_train)
svr_test_score=svr.score(X_test, y_test)
f.write("SVN_train:{}".format(svr_train_score))
f.write("SVN_test:{}".format(svr_test_score))
try:
	f.write("{}".format(svr.best_estimator_)) 
except:
	print "except"

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3,
                  param_grid={"alpha": [1e0, 0.1, 1.0e-3],
                              "gamma": np.logspace(-2, 2, 5)})

kr.fit(X_train, y_train)
kr_train_score=kr.score(X_train, y_train)
kr_test_score=kr.score(X_test, y_test)
f.write("kr_train:{}".format(kr_train_score))
f.write("kr_test:{}".format(kr_test_score))
try:
	f.write("{}".format(kf.best_estimator_)) 
except:
	print "except"

rf=GridSearchCV(RandomForestRegressor(n_estimators=100)	, cv=3, param_grid=	{'max_depth':[4,6,10]})


rf.fit(X_train, y_train)
rf_train_score=rf.score(X_train, y_train)
rf_test_score=rf.score(X_test, y_test)
f.write("randomforest_train:{}".format(rf_train_score))
f.write("randomforest_test:{}".format(rf_test_score))
try:
	f.write("{}".format(rf.best_estimator_)) 
except:
	print "except"


pdb.set_trace()

print svr_train_score, svr_test_score 
print kr_train_score, kr_test_score 
print rf_train_score, rf_test_score

pdb.set_trace()

# for df in [t_menu,t_name,test1]:
# 	X = df
# 	y = df['price'].values
# 	svr.fit(X,y)
	# X_train, X_test, y_train, y_test = train_test_split(X, y)

	# cv = KFold(3)
	# ridge = RidgeCV(cv=cv)
	# ridge.fit(X_train, y_train)
	# R_ridge_train_score = ridge.score(X_train, y_train)
	# R_ridge_test_score = ridge.score(X_test, y_test)
	# y_pred_ridge = ridge.predict(X_test)

	# print R_ridge_train_score, R_ridge_test_score

