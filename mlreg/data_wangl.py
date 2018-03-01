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
test_df = data[['restaurant_id','city_id','postcode','menu_category','product_name', 'price']].copy()
test_df.dropna(subset=['menu_category','product_name','city_id'], inplace=True)
print test_df.isnull().sum()
print 'Unique restaurants: {}'.format(len(data['restaurant_id'].unique()))
print 'Unique menu_category: {}'.format(len(data['menu_category'].unique()))
print 'Unique product_name: {}'.format(len(data['product_name'].unique()))
print 'Unique ingredients: {}'.format(len(data['ingredients'].unique()))
print test_df.shape

# encode_menu=test_df['menu_category'].str.encode('ascii', errors='ignore')
# encode_menu.replace({r'[^a-zA-Z0-9\s,]':''}, regex=True, inplace=True)
# encode_name=test_df['product_name'].str.encode('ascii', errors='ignore')
# encode_name.replace({r'[^a-zA-Z0-9\s,]':''}, regex=True, inplace=True)

# pdb.set_trace()
# all_combinations = set(np.concatenate((encode_menu.unique(), encode_name.unique())))

# words = set([el for word in all_combinations for el in str(word).split()])
# # words=set([x.strip() for el in all_words.values() for x in str(el).split()])
# # dict_all=[PorterStemmer().stem(str(x)) for x in words.unique()]
# dict_all = set([GermanStemmer().stem(x) for x in words])

# pdb.set_trace()



# tfidf_vectorizer = TfidfVectorizer(max_features=n_features)
# tfidf = tfidf_vectorizer.fit_transform(dict_all)

# Use tf (raw term count) features for LDA.
# print("Extracting tf features for LDA...")
# tf_vectorizer = CountVectorizer(max_features=n_features)
# tf = tf_vectorizer.fit_transform(dict_all)

# # Fit the NMF model
# print("Fitting the NMF model (Frobenius norm) with tf-idf features")
# nmf = NMF(n_components=n_components,random_state=1,
#           alpha=.1, l1_ratio=.5).fit(tfidf)

# print("\nTopics in NMF model (Frobenius norm):")
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print_top_words(nmf, tfidf_feature_names, 100)

# # Fit the NMF model
# print("Fitting the NMF model (generalized Kullback-Leibler divergence)")
# nmf = NMF(n_components=n_components, random_state=1,
#           beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
#           l1_ratio=.5).fit(tfidf)

# print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print_top_words(nmf, tfidf_feature_names, 100)

# print("Fitting LDA models with tf features, ")
# lda = LatentDirichletAllocation(max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)

# lda.fit(tf)

# print("\nTopics in LDA model:")
# tf_feature_names = tf_vectorizer.get_feature_names()
# print_top_words(lda, tf_feature_names, 20)



pdb.set_trace()

# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w')
# z=tokenizer.tokenize(test_df['menu_category'].unstack())
# pdb.set_trace()
# word_vectorizer = CountVectorizer(analyzer='char',strip_accents='ascii')

word_vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='char',strip_accents='ascii', token_pattern='\w|,|')
sparse_matrix_menu = word_vectorizer.fit_transform(test_df['menu_category'])
# frequencies = sum(sparse_matrix).toarray()[0]
# new=pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
col_names = ['menu_' + el for el in word_vectorizer.get_feature_names()]
print 'Added {} features for the menu'.format(len(col_names))
new_features_menu=pd.DataFrame(sparse_matrix_menu.todense(), index=test_df.index, columns=col_names)

sparse_matrix_name = word_vectorizer.fit_transform(test_df['product_name'])
col_names = ['name_' + el for el in word_vectorizer.get_feature_names()	]
print 'Added {} features for the product name'.format(len(col_names))
new_features_name=pd.DataFrame(sparse_matrix_name.todense(), index=test_df.index, columns=col_names)

test_df['restaurant_id'] = test_df['restaurant_id'].get_dummies()	
X =	pd.concat([new_features_name, new_features_menu, test_df[['restaurant_id','city_id']]], axis=1)
pdb.set_trace()
y = test_df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

estimator=RandomForestRegressor(n_estimators=100)

param_grid={
	'n_estimators':[20, 50, 100], 
	#'learning_rate': [0.1,0.05, 0.02, 0.01],
	'max_depth':[4,6, 10], 
	'min_samples_leaf':[3,5,9,17], 
	'max_features':[1.0,0.5,0.1] 
	} 

n_jobs = -1
cv = KFold(3)
ridge = RidgeCV(cv=cv)
ridge.fit(X_train, y_train)
R_ridge_train_score = ridge.score(X_train, y_train)
R_ridge_test_score = ridge.score(X_test, y_test)
y_pred_ridge = ridge.predict(X_test)
# regressor = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
# regressor.fit(X_train, y_train) 
for k, (train, test) in enumerate(cv.split(X, y)):
	pdb.set_trace()
 	estimator.fit(X[train], y[train]) 
 	y_pred=estimator.predict(X_test) 

# # best_est= regressor.best_estimator_ 

# # print "Best Estimator Parameters" 
# # print"---------------------------" 
# # print "n_estimators: %d" %best_est.n_estimators 
# # print "max_depth: %d" %best_est.max_depth 
# # print "Learning Rate: %.1f" %best_est.learning_rate 
# # print "min_samples_leaf: %d" %best_est.min_samples_leaf 
# # print "max_features: %.1f" %best_est.max_features 
# # print "Train R-squared: %.2f" %best_est.score(X_train,y_train) 
	print "Feature Importances" 
	print estimator.feature_importances_ 
	print "R-squared for Train: %.2f" %estimator.score(X[train], y[train]) 
	print "R-squared for Test: %.2f" %estimator.score(X[test], y[test])


dummy = DummyRegressor()
dummy.fit(X_train, y_train)
R_dummy=dummy.score(X_train, y_train) 
y_pred_dummy = dummy.predict(X_test)

pdb.set_trace()