import numpy as np

class epd:

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def getPosition(self, sent):
		l = len(sent)

		#random sample from normal distribution
		s = -1
		while s < 0:
			s = np.random.normal(loc=self.mean, scale=self.std)
			s = np.fix(s)

		#random int 0 or 1, 0=from beginning of sentence, 1=from end
		r = np.random.randint(2)

		if r == 0:
			pos = s

		else:
			pos = l-s - 1

		if pos < 0 or pos > l:
			return None

		else:
			return pos

