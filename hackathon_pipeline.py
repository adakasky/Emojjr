from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import nltk
from nltk.tokenize import word_tokenize
import re
import epd
import numpy as np

tags = set(["NN", "JJR", "JJS" "NNS", "NNP", "NNPS" "JJ", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
def getTag(lis, index, returnType=False):
	if returnType == True:
		return False
	else:
		if index == None:
			return False
		#index = int(index - 1)
		if lis[index][1] in tags:
			return True
		else:
			return False


emoji = {u'\ud83d\ude00': [30, 30, 40, 40, 30],
u'\ud83d\ude2c': [-10, 25, -45, 10, -30],
u'\ud83d\ude01': [50, 22, 45, 50, 40],
u'\ud83d\ude02': [40, 42, 50, -20, 45],
u'\ud83d\ude03': [40, 24, 45, 30, 35],
u'\ud83d\ude05': [10, 13, 50, -10, 20],
u'\ud83d\ude06': [50, 33, 45, 0, 45],
u'\ud83d\ude07': [20, 17, 20, 40, 30],
u'\ud83d\ude09': [25, 36, 15, 40, 20],
u'\ud83d\ude0a': [40, 23, 30, 40, 40],
u'\ud83d\ude43': [-20, 7, 25, 40, 50],
u'\ud83d\ude42': [20, 10, 10, 20, 35],
u'\ud83d\ude0c': [40, 18, 5, 20, 38],
u'\ud83d\ude0d': [10, 49, 40, 50, 48],
u'\ud83d\ude18': [10, 38, 0, 40, 49],
u'\ud83d\ude17': [0, 28, 30, 20, 15],
u'\ud83d\ude19': [40, 32, -5, 30, -5],
u'\ud83d\ude1c': [20, 44, 45, 50, 10],
u'\ud83d\ude1a': [40, 41, -10, 40, 29],
u'\ud83e\udd11': [-20, 45, 40, 50, -10],
u'\ud83d\ude0e': [50, 50, 10, 50, 50],
u'\ud83e\udd17': [30, 19, 25, 50, -10],
u'\ud83d\ude0f': [-25, 20, -25, 40, 32],
u'\ud83d\ude36': [-30, 0, -30, 0, 20],
u'\ud83d\ude10': [-40, -6, 20, 0, 10],
u'\ud83d\ude11': [-50, -10, 0, -5, 45],
u'\ud83d\ude12': [-50, -15, -20, -40, 47],
u'\ud83d\ude44': [-50, -6, -40, 0, 50],
u'\ud83e\udd14': [0, -3, -20, 0, 39],
u'\ud83d\ude33': [0, -5, 10, 10, 26],
u'\ud83d\ude1e': [-50, -20, -10, -20, 5],
u'\ud83d\ude20': [-50, -40, -45, -40, 20],
u'\ud83d\ude21': [-50, -50, -30, -50, 23],
u'\ud83d\ude14': [-30, -25, -35, -30, -1],
u'\ud83d\ude15': [-25, -14, -40, -10, 13],
u'\ud83d\ude16': [-40, -48, -10, -20, -20],
u'\ud83d\ude2b': [-20, -47, -35, -30, 5],
u'\ud83d\ude29': [0, -31, -25, -30, 0],
u'\ud83d\ude24': [-50, -50, 15, -40, 15],
u'\ud83d\ude2e': [0, 2, 0, 0, 0],
u'\ud83d\ude31': [0, -39, -5, -50, 40],
u'\ud83d\ude28': [-40, -34, -10, 0, 30],
u'\ud83d\ude30': [-50, -38, 20, -30, 43],
u'\ud83d\ude26': [0, -22, -50, -20, 9],
u'\ud83d\ude22': [-10, -25, -45, -30, 35],
u'\ud83d\ude2a': [-25, -12, 0, -20, 14],
u'\ud83d\ude2d': [0, -36, -50, -50, 47],
u'\ud83d\ude35': [0, -42, 10, -10, 8],
u'\ud83d\ude37': [-40, -17, -20, -40, 20],
u'\ud83d\ude34': [0, 5, -30, -10, 48],
u'\ud83d\udca9': [0, 16, -50, 10, 39],
u'\ud83d\ude08': [0, 3, -45, 10, 19],
u'\ud83d\udc7f': [-20, -3, -40, -10, 8],
u'\ud83d\udc7b': [25, 47, 0, 30, 47],
u'\ud83d\udc80': [-20, -9, -50, -20, 35],
u'\ud83d\udc7d': [20, 18, 10, 20, 19],
u'\ud83d\udc7e': [20, 20, 20, -10, 23],
u'\ud83d\ude3a': [40, 43, 40, 10, 5],
u'\ud83d\ude39': [40, 45, 45, 0, 3],
u'\ud83d\ude3b': [-10, 49, 35, 50, 5],
u'\ud83d\ude3f': [-20, -20, -50, -30, 10],
u'\ud83d\udc4d': [25, 27, 25, 50, 50],
u'\ud83d\udc4e': [-25, -27, -25, -50, 35],
u'\ud83d\udc4f': [30, 14, 30, 50, 40],
u'\ud83d\udc4c': [50, 46, 25, 30, 50],
u'\u270c': [25, 12, 0, 20, 50],
u'\ufe0f\ud83d\ude4f': [10, 14, -10, 0, 40],
u'\ud83d\udcaa': [5, 50, 20, 20, 42],
u'\ud83d\udc40': [-30, 20, 40, 20, 48],
u'\ud83d\udc45': [-40, 2, 45, 30, -5],
u'\ud83d\udc44': [-50, 5, 20, 40, 12],
u'\ud83d\udc85': [0, 10, -15, 0, 17],
u'\ud83d\ude45': [-20, -10, -30, -30, 20],
u'\u200d\ud83d\ude46': [0, 10, 30, 10, 0],
u'\ud83d\udc81': [-10, -8, -20, 20, 39],
u'\ud83d\udc83': [30, 20, 10, 50, 20],
u'\ud83d\udc6f': [40, 22, 10, 40, 29],
u'\ud83d\udeb6': [-5, -3, 0, -30, 5],
u'\ud83d\udc69': [5, 10, 0, 20, 30],
u'\ud83d\udc51': [-5, 40, 40, 30, 28],
u'\ud83d\udc36': [50, 43, 20, 30, 40],
u'\ud83d\udc31': [50, 44, 30, 20, 36],
u'\ud83d\udc2d': [50, 32, -20, 0, 20],
u'\ud83d\udc37': [0, 35, -30, 20, 29],
u'\ud83d\udc38': [10, 31, -10, 0, 9],
u'\ud83d\udc19': [-10, 36, -40, -10, 10],
u'\ud83d\udc35': [5, 38, 20, 30, 12],
u'\ud83d\ude48': [20, 20, 30, -10, 26],
u'\ud83d\ude49': [0, 19, 30, -10, 15],
u'\ud83d\ude4a': [-10, 18, 30, -10, 39],
u'\ud83d\udc34': [50, 30, 50, 10, 7],
u'\ud83d\udc26': [0, 33, 0, 0, 8],
u'\ud83d\udc1d': [0, 50, 25, 0, 45],
u'\ud83d\udc22': [10, 32, 0, -10, 20],
u'\ud83d\udc33': [50, 37, 25, 10, 40],
u'\ud83d\udc0d': [-20, 13, -20, 0, 20],
u'\ud83c\udf35': [0, -10, -30, -10, 37],
u'\ud83c\udf41': [-10, 42, 10, 0, 25],
u'\ud83c\udf0e': [0, 12, 0, 10, 39],
u'\ud83c\udf1d': [-10, 34, -20, 20, 49],
u'\ud83c\udf1a': [-10, 31, 20, 10, 50],
u'\ud83c\udf1e': [-20, 33, 10, 40, 30],
u'\ud83c\udf1b': [-10, 19, 10, 20, 15],
u'\ud83c\udf1c': [-10, 29, -10, 20, 15],
u'\ud83c\udf1f': [20, 20, 40, 30, 36],
u'\ud83d\udcab': [30, 24, 35, 0, 40],
u'\ud83d\udca6': [-25, 10, 30, -10, 20],
u'\u26c4': [20, 30, 20, 30, 46],
u'\ufe0f\ud83d\udd25': [50, 44, 50, 20, 45],
u'\u26a1': [40, 42, 45, -10, 47],
u'\ufe0f\ud83c\udf46': [50, 20, 50, 0, 44],
u'\ud83c\udf51': [50, 26, 30, 10, 50],
u'\ud83c\udf52': [25, 29, 30, 10, 49],
u'\ud83c\udf4c': [0, 10, 40, 10, 47],
u'\ud83c\udf3d': [0, 16, 10, 10, 37],
u'\ud83c\udf64': [-10, 20, 20, 20, 40],
u'\ud83c\udf7a': [20, 24, 10, 30, 13],
u'\ud83c\udf7b': [30, 25, 10, 40, 15],
u'\ud83c\udf7e': [40, 26, 10, 50, 50],
u'\ud83c\udf77': [20, 27, 20, 10, 41],
u'\ud83c\udf78': [20, 20, 15, 20, 20],
u'\ud83c\udf79': [20, 22, 0, 30, 25],
u'\u2615': [10, 14, -30, 20, 43],
u'\ufe0f\u26bd': [30, 0, -15, 10, 10],
u'\ud83c\udfc0': [30, 0, 10, 10, 15],
u'\ud83c\udfc8': [50, 0, 10, 10, 29],
u'\u26be': [10, 0, 20, 10, 20],
u'\ud83c\udfc7': [50, 2, 15, 20, -6],
u'\ud83c\udfc6': [40, 37, 40, 50, 4],
u'\ud83d\ude97': [20, 21, -10, 10, -10],
u'\ud83d\ude80': [30, 46, -30, 30, 43],
u'\ud83c\udf08': [50, 43, 30, 40, 49],
u'\ud83d\udcfc': [20, 22, 0, 0, 0],
u'\ud83d\udcb8': [50, 30, -20, 40, 38],
u'\ud83d\udcb0': [30, 33, 30, 45, 30],
u'\ud83d\udc8e': [20, 40, 34, 50, 40],
u'\ud83d\udd27': [0, 49, -10, -10, 9],
u'\ud83d\udca3': [25, 20, -20, -50, 2],
u'\ud83d\udd2b': [-40, -3, -40, -30, -8],
u'\ud83d\udebd': [30, -15, -50, -10, 10],
u'\ud83d\uddff': [0, 29, 30, 0, 48],
u'\ud83c\udf89': [40, 39, 40, 50, 45],
u'\ud83d\udcce': [0, -2, 10, 0, 18],
u'\ud83c\udff3': [-20, -9, 20, -50, 5],
u'\ufe0f\ud83c\udf08': [50, 14, 30, 50, 49],
u'\u2764': [50, 2, 10, -30, 46],
u'\ufe0f\ud83d\udc9b': [49, 10, -10, 30, 40],
u'\ud83d\udc9a': [48, 14, -20, 20, 44],
u'\ud83d\udc99': [47, 8, 20, -10, 50],
u'\ud83d\udc9c': [46, 29, 25, 20, 49],
u'\ud83d\udc95': [45, 40, 10, 50, 50],
u'\ud83d\udc94': [-50, -28, 20, -50, 40],
u'\u274c': [-45, -50, -50, -20, 30],
u'\u2b55': [25, 50, 50, 20, 0],
u'\ufe0f\u2668': [15, 39, 50, 20, 46],
u'\ufe0f\ud83d\udcaf': [50, 48, 50, 50, 50],
u'\ud83c\udd97': [50, 15, 0, 10, 40],
u'\ud83c\udd92': [50, 29, 25, 30, 47],
u'\u00ae': [50, 0, -20, 0, -8],
u'\ud83d\udd14': [50, 3, 20, 0, -1],
u'\ud83c\udde8\ud83c\udde6': [30, 15, 25, 0, 20],
u'\ud83c\uddec\ud83c\udde7': [30, 15, -35, 0, 15],
u'\ud83c\uddfa\ud83c\uddf8': [50, 50, 45, 50, 38]}

def func(std, emojificationParam, inFile, outFile="out.txt"):

	with open(inFile, "r") as corpus:
		with open(outFile, "w") as output:
			content = corpus.read()
			sentences = re.split("[?\.!]", content)
			for sentence in sentences:
			#for i in range(0, 5):
				sentence = sentences[i]
				vs = vaderSentiment(sentence)
				#print "\t" + str(vs)
				text = word_tokenize(sentence)
				pos_tags = nltk.pos_tag(text)
				myEpd = epd.epd(0, std)
				emojipos = []
				for sample in range(0, int(np.random.rand() * emojificationParam * len(text))):
					emojipos+=[int(myEpd.getPosition(text))]
				righttags = []
				for position in emojipos:
					if getTag(pos_tags, position) == True:
						righttags+=[position]
				score = vs["compound"] * 50
				final_emojis = []
				emojis = []#index to list of unicode values 
				for k,v in emoji.items(): 
					for val in v:
						if val >= score - 15 and val <= score + 15:
							emojis+= [k]
				for index in righttags:
					sample = int(np.floor(np.random.rand() *len(emojis)))
					choice = emojis[sample]
					final_emojis+= [(index, choice)]
				text = [unicode(t, "utf-8") for t in text]
				for index,e in final_emojis:
					if index == 0:
						samp = np.random.rand()
						if samp > .5:
							text[index] = text[index] + e
						else:
							text[index] = e+  text[index]
					else:
						text[index] = text[index] + e
				output.write(" ".join(text).encode("utf-8"))
				print " ".join(text)


	corpus.close()
	output.close()

if __name__ == '__main__':

	func(2, 2, "imagenetcorp.txt")
