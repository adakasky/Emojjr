from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk.internals import find_jars_within_path
import time
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.tag import StanfordPOSTagger

jar = "/home/aaron/Downloads/stanford-postagger-2015-12-09/stanford-postagger.jar"
model = "/home/aaron/Downloads/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger"

with open("imagenetcorp.txt", "r") as corpus:
	content = corpus.read()
	sentences = re.split("[?\.!]", content)
	#for sentence in sentences:
	for i in range(0, 5):
		sentence = sentences[i]
		vs = vaderSentiment(sentence)
		print "\t" + str(vs)
		toc = time.time()
		text = word_tokenize(sentence)
		tags = nltk.pos_tag(text)
		print tags
corpus.close()

tags = set("NN", "JJR", "JJS" "NNS", "NNP", "NNPS" "JJ", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
def getTag(lis, index, returnType=False):
	if returnType:
		return False
	else:
		if lis[index][1] in tags:
			return True
		else:
			return False
