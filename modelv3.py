import sklearn
#from pprint import pprint
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline

# Get paths to labelled data
rawFolderPaths = glob("/Users/danielhoadley/PycharmProjects/trainer/!labelled_data_reportXML/*/")

print ('\nGathering labelled categories...\n')

# categories = ['crime', 'contract', 'negligence', 'judicial review', 'insolvency', 'defamation', 'employment', 'evidence', 'extradition', 'commonwealth']

categories = []

# Extract the folder paths, reduce down to the label and append to the categories list
for i in rawFolderPaths:
    string1 = i.replace('/Users/danielhoadley/PycharmProjects/trainer/!labelled_data_reportXML/','')
    category = string1.strip('/')
    #print (category)
    categories.append(category)

# Load the data
print ('\nLoading the dataset...\n')
docs_to_train = sklearn.datasets.load_files("/Users/danielhoadley/PycharmProjects/trainer/!labelled_data_reportXML",
                                            description=None, categories=categories, load_content=True,
                                            encoding='utf-8', shuffle=True, random_state=42)

# Split the dataset into training and testing sets
print ('\nBuilding out hold-out test sample...\n')
X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data, docs_to_train.target, test_size=0.4)

# THE TRAINING DATA

# Transform the training data into tfidf vectors
print ('\nTransforming the training data...\n')
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(raw_documents=X_train)

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#print (X_train_tf.shape)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print (X_train_tfidf.shape)

# THE TRAINING DATA

# Transform the test data into tfidf vectors
print ('\nTransforming the test data...\n')
count_vect = CountVectorizer(stop_words='english')
X_test_counts = count_vect.fit_transform(raw_documents=X_test)

###tf_transformer = TfidfTransformer(use_idf=False).fit(X_test_counts)
##X_test_tf = tf_transformer.transform(X_test_counts)
#print (X_test_tf.shape)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
print (X_test_tfidf.shape)

print (X_test_tfidf)
print (y_train.shape)

docs_test = X_test

# Construct the classifier pipeline using a SGDClassifier algorithm
print ('\nApplying the classifier...\n')
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42, verbose=1)),
])

# Fit the model to the training data
text_clf.fit(X_train, y_train)

# Run the test data into the model
predicted = text_clf.predict(docs_test)

# Calculate mean accuracy of predictions
print (np.mean(predicted == y_test))

# Generate labelled performance metrics
print(metrics.classification_report(y_test, predicted,
    target_names=docs_to_train.target_names))










