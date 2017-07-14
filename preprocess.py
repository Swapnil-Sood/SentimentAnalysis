import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
	
def test_data():
	tweets = open('test_tweets.txt','w')
	polarity =open('test_polarity.txt','w')
	with open('twitter-2016test-A.txt') as fi:
		for line in fi:
			pol = line.split('\t')[1]
			tweet = line.split('\t')[-2]
			tweet = tweet.translate(str.maketrans({'"':None}))
			tweets.write(str(tweet))
			tweets.write('\n')
			polarity.write(pol)
			polarity.write('\n')
test_data()   
	
def val_data():
	tweets= open('val_tweets.txt','w')
	polarity = open('val_polarity.txt','w')
	with open('twitter-2016dev-A.txt') as fi:
	    for line in fi:
	        pol = line.split('\t')[1]
	        tweet = line.split('\t')[-1]
	        tweets.write(tweet)
	        polarity.write(pol)
	        polarity.write('\n')
val_data()


def shufle():
	lines = open('twitter-2016train-A.txt').readlines()
	random.shuffle(lines)
	open('datafile.txt', 'w').writelines(lines)


shufle()
def testdata():
	a= [x.split('\t')[2] for x in open('datafile.txt').readlines()]
	f = open('tweets.txt','w')
	for item in a:
		f.write("%s"%item)
testdata()

def test_polarity():
	a= [x.split('\t')[1] for x in open('datafile.txt').readlines()]
	f = open('polarity.txt','w')
	for item in a:
		f.write("%s\n"%item)
test_polarity()


def loadGloveModel(gloveFile='glove.twitter.27B.50d.txt'):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    i=0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
        i+=1
        if(i==100000):break
    print("Done.",len(model)," words loaded!")
    return model
  

def maxline(model):
	n=0
	with open('tweets.txt','r') as f:
		contents = f.readlines()
		for line in contents:
			no_of_words =0
			words = list(word_tokenize(line.lower()))
			for word in words:
				if word in model:
					no_of_words +=1
			n = max(n,no_of_words)
	return n


 
def fn(n,model,d,tweet_file,pol_file):

	featureset = []
	pol = open(pol_file,'r')
	p = list(pol)
	i=0
	
	with open(tweet_file,'r') as f:
		contents = f.readlines()
		
		for line in contents:
			featureset = list(featureset)
			features = []
			no_of_words =0
			words = list(word_tokenize(line.lower()))
			for word in words:
				if word in model:
					features.append(model[word])
					features = list(features)
					no_of_words +=1
					if(no_of_words == n):break
			while(no_of_words<n):
				a = [0 for y in range(d)]
				features.append(a)
				features = list(features)
				no_of_words += 1

			
			if (p[i] == 'positive\n'):
				featureset.append([features,[1, 0, 0]])
			elif (p[i]=='neutral\n'):
				featureset.append([features,[0, 1, 0]])
			elif (p[i]=='negative\n'):
				featureset.append([features,[0, 0, 1]])
			i = i+1
	return featureset
def fn_glove():
	model_glove = loadGloveModel()
	n_glove = maxline(model_glove)
	print("n_glove:",n_glove)
	
	
	features_glove = []
	features_glove += fn(n_glove,model_glove,50,'tweets.txt','polarity.txt')
	features_glove = np.array(features_glove)
	print("No of train tweets:",len(features_glove))

	train_xg = list(features_glove[:,0])
	train_yg = list(features_glove[:,1])

	features_glove = []
	features_glove += fn(n_glove,model_glove,50,'val_tweets.txt','val_polarity.txt')
	features_glove = np.array(features_glove)
	print("No of val tweets:",len(features_glove))

	val_xg = list(features_glove[:,0])
	val_yg = list(features_glove[:,1])

	features_glove = []
	features_glove += fn(n_glove,model_glove,50,'test_tweets.txt','test_polarity.txt')	
	features_glove = np.array(features_glove)
	print("No of test tweets:",len(features_glove))
	test_xg = list(features_glove[:,0])
	test_yg = list(features_glove[:,1])
	return train_xg,train_yg,val_xg,val_yg,test_xg,test_yg