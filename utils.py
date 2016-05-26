"""
    Utilities
"""

import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_sentences(train):
	result = []
	for rev in train:
		for s in rev:
			x, y = zip(*s)
			result.append(x)
	return result

def get_X(train):
	result = []
	for rev in train:
		sentences = []
		for s in rev:
			x, y = zip(*s)
			sentences.append(x)
		result.append(sentences)
	return result

def build_vocabulary(sentences):
	"""
		Build a mapping between every word and 
	"""
	words = [""]
	for sentence in sentences:
		for word in sentence:
			words.append(word.lower())
	words = sorted(set(words))
	print([(x,i) for i,x in enumerate(words)])
	vocabulary = {x: i for i, x in enumerate(words)}

	return vocabulary


def format_data(reviews,target,vocabulary,length):

	#Mapping words with values in the vocabulary
	x = np.array([[[vocabulary[word.lower()] for word in sentence] for sentence in review] for review in reviews])

	x = np.array(fromReviewsToInput(x,length))
	#y = np.array(target)

	#return [x,y]

def fromReviewsToInput(x,length):
	result = []
	for r in x:
		#
		concatenatedReview = []
		for s in r:
			concatenatedReview += s
		print(concatenatedReview)
		result += generateInput(concatenatedReview, length)
	return result

def generateInput(concatenatedReview,length):
	result = []
	i = 0
	for i in range(len(concatenatedReview)):
		partialResult = []
		blankBegin = 0
		blankEnd = 0
		if i < length -1:
			blankBegin = length - 1 - i
			for j in range(blankBegin):
				partialResult.append(0) # 0 is the empty word ""
		
		for j in range(i,min(i+length-blankBegin,len(concatenatedReview))):
			partialResult.append(concatenatedReview[j])
		
		if i + length > len(concatenatedReview):
			blankEnd = i + length - len(concatenatedReview)
			for j in range(blankEnd):
				partialResult.append(0) # 0 is the empty word ""
		result.append(partialResult)
	return result