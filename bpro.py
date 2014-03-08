import argparse
import traceback
import glob
from utils_stats import FrequencyAssessor
import nltk
import string
import random
import json
from utils_io import Yamlator
from sklearn.cross_validation import train_test_split
import traceback
from sklearn import cross_validation
from random import shuffle
import sklearn
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier

class Admin(object):
    """
    Main client for modeling customer profile data with text.
    """
    def main(self):


        parser = argparse.ArgumentParser(description='modeling customer profile data with text')
        parser.add_argument('-g', action='store_true', help="generate customer and behavioral profile data")
        parser.add_argument('-a', action='store_true', help="analyze behavioral profile data")

        args = parser.parse_args()

        try:
            if args.g:
                generate_profile_data(100,50,2,10,100)
            elif args.a:
                behavioral_profiles = Yamlator.load("behavioral_profiles.yaml")
                customers = Yamlator.load("customers.yaml")
                analyze(behavioral_profiles, customers)

        except:
            traceback.print_exc()

def analyze(behavioral_profiles, customers):
    """
    Analyze profile data.
    """

    #build behavioral profile models.

    #first build up a corpus representation of product descriptions.
    vectorizer = CountVectorizer(min_df=1)

    corpus=[]
    for bp in behavioral_profiles:
        for pd in bp.product_descriptions:
            corpus.append(pd.description)


    #Tokenize the product descriptions into individual words and build a dictionary of terms. Each term found by the
    # analyzer during the fitting process is given a unique integer index that corresponds to a column in the
    # resulting matrix. Note: this tokenizer configuration also drops single character words.

    vectorizer.fit_transform(corpus)

    print vectorizer.get_feature_names()[200:210]
    #print vectorizer.transform(['pink clouds']).toarray()[0]

    #Randomize the observations
    data_target_tuples=[]
    for bp in behavioral_profiles:
        for pd in bp.product_descriptions:
            data_target_tuples.append((bp.type, pd.description))

    shuffle(data_target_tuples)

    #Build the observation feature and target vectors
    X_data=[]
    y_target=[]
    for t in data_target_tuples:
        v = vectorizer.transform([t[1]]).toarray()[0]
        X_data.append(v)
        y_target.append(t[0])

    X_data=np.asarray(X_data)
    y_target=np.asarray(y_target)

    #evaluate model to make sure it is reasonable on the behavioral profile data
    linear_svm_classifier = SVC(kernel="linear", C=0.025)
    scores = sklearn.cross_validation.cross_val_score(OneVsRestClassifier(linear_svm_classifier), X_data, y_target, cv=2)
    print("Accuracy using %s: %0.2f (+/- %0.2f) and %d folds" % ("Linear SVM", scores.mean(), scores.std() * 2, 5))


    #Do a full training of the model
    behavioral_profiler = SVC(kernel="linear", C=0.025)
    behavioral_profiler.fit(X_data, y_target)

    #Take it out for a spin
    print behavioral_profiler.predict(vectorizer.transform(['Some black shoes to go with your Joy Division hand bag']).toarray()[0])
    print behavioral_profiler.predict(vectorizer.transform(['Ozzy Ozbourne poster, 33in x 24in']).toarray()[0])

    #Now on to classifying our customers
    predicted_profiles=[]
    ground_truth=[]
    for c in customers:
        customer_product_descriptions = ' '.join(p.description for p in c.product_descriptions)
        predicted = behavioral_profiler.predict(vectorizer.transform([customer_product_descriptions]).toarray()[0])
        predicted_profiles.append(predicted[0])
        ground_truth.append(c.type)
        print "Customer %d, known to be %s, was predicted to be %s" % (c.id,c.type,predicted[0])

    #print predicted_profiles
    #print ground_truth

    a=[x1==y1 for x1, y1 in zip(predicted_profiles,ground_truth)]
    accuracy=float(sum(a))/len(a)
    print "Percent Profiled Correctly %.2f" % accuracy

def generate_profile_data(num_customer_profiles,
                            terms_per_profile,
                            min_products_purchased_per_customer,
                            max_products_purchased_per_customer,
                            num_products_per_behavioral_profile):
    """
    Generates customer and behavioral profile data.
    """

    #builds term/count map by genre
    genre_freq_ass_map=dict()
    print "building user product data set"
    for f in glob.glob("seed/*.txt"):
        genre = f.split("/")[1].split("-")[0]
        if genre not in genre_freq_ass_map:
            genre_freq_ass_map[genre]=FrequencyAssessor()

        with open (f, "r") as current_file:
            text = current_file.read()
            tokens = nltk.word_tokenize(text)
            for t in tokens:
                if t not in string.punctuation and len(t) > 2:
                    if t.endswith("."):
                        t=t[:len(t)-1]

                    genre_freq_ass_map[genre].update(t.lower())

    #build genre>list of terms freq of occurance weighted.
    genre_terms_list=dict()
    for g in genre_freq_ass_map.keys():
        genre_terms_list[g]=[]
        for t in genre_freq_ass_map[g].get_top_terms(max=500):
            for i in range(t[1]):
                genre_terms_list[g].append(t[0])

    #build customer profiles
    customer_profiles=[]
    for g in genre_terms_list.keys():
        cid=0
        for c in range(num_customer_profiles):
            product_descriptions=[]
            pid=0
            for p in range(random.randint(min_products_purchased_per_customer,
                                          max_products_purchased_per_customer)):
                product_description=[]
                for i in range(terms_per_profile):
                    word_index = random.randint(0,len(genre_terms_list[g])-1)
                    product_description.append(genre_terms_list[g][word_index])

                    z=0
                    while z<4:
                        #pop in some random words
                        ii = random.randint(0,len(genre_terms_list.keys())-1)
                        random_genre = genre_terms_list.keys()[ii]
                        word_index = random.randint(0,len(genre_terms_list[random_genre])-1)
                        product_description.append(genre_terms_list[random_genre][word_index])
                        z+=1
                pid+=1
                product_descriptions.append(Product(pid, ' '.join(product_description)))
            cid+=1
            customer_profiles.append(Customer(g, cid, product_descriptions))

    Yamlator.dump("customers.yaml", customer_profiles)

    #build behavioral_profiles
    behavioral_profiles=[]
    for g in genre_terms_list.keys():
        product_descriptions=[]
        pid=0
        for p in range(num_products_per_behavioral_profile):
            product_description=[]
            for i in range(terms_per_profile):
                word_index = random.randint(0,len(genre_terms_list[g])-1)
                product_description.append(genre_terms_list[g][word_index])

            pid+=1
            product_descriptions.append(Product(pid, ' '.join(product_description)))

        behavioral_profiles.append(BehavioralProfile(g, product_descriptions))

    Yamlator.dump("behavioral_profiles.yaml", behavioral_profiles)

class Customer(object):

    def __init__(self, type, id, product_descriptions):
        self.type=type
        self.id=id
        self.product_descriptions=product_descriptions

class Product(object):

    def __init__(self, id, description):
        self.id=id
        self.description=str(unicode(description, errors='ignore'))

class BehavioralProfile(object):

    def __init__(self, type, product_descriptions):
        self.type=type
        self.product_descriptions=product_descriptions

if __name__ == "__main__":
    Admin().main()

