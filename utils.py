"""
    Utilities
"""

import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=False):
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

def get_formatted_sentences	(corpus,vocabulary,length):
	result = []
	result_y = []
	for rev in corpus:
		sentences = []
		sentences_y = []
		for s in rev:
			x1, x2 = zip(*s)
			sentences.append(x1)
			sentences_y.append(x2)
		result.append(sentences)
		result_y.append(sentences_y)


	x,y = format_data(result,result_y,vocabulary,length)

	return [x, y]

def build_vocabulary(sentences):
	"""
		Build a mapping between every word and 
	"""
	words = [""]
	for sentence in sentences:
		for word in sentence:
			words.append(word)
	words = sorted(set(words))
	#print([(x,i) for i,x in enumerate(words)])
	vocabulary = {x: i for i, x in enumerate(words)}

	return vocabulary


def format_data(reviews,target,vocabulary,length):

	#Mapping words with values in the vocabulary
	x = np.array([[[vocabulary[word] for word in sentence] for sentence in review] for review in reviews])

	x = np.array(fromReviewsToInput(x,length))

	y = np.array([[[ word_target for word_target in sentence_target] for sentence_target in t] for t in target])
	
	
	y_result = []
	for s in y:
		for w in s:
			y_result += w
	print(len(x))
	print(len(y_result))

	y = []
	for w in y_result:
		m = [0]*3
		m[w] = 1
		y.append(m)
	return [x,y]

def fromReviewsToInput(x,length):
	result = []
	for r in x:
		#
		concatenatedReview = []
		for s in r:
			concatenatedReview += s
		result += generateInput(concatenatedReview, length)
	return result

def generateInput(concatenatedReview,length):
	result = []
	i = 0
	for i in range(len(concatenatedReview)):
		partialResult = []
		blankBegin = 0
		blankEnd = 0
		if i < length/2:
			blankBegin = length/2 - i
			for j in range(blankBegin):
				partialResult.append(0) # 0 is the empty word ""
		
		for j in range(max(	i-length/2,0),min(i+length/2+1,len(concatenatedReview))):
			partialResult.append(concatenatedReview[j])
		
		if i + length/2 >= len(concatenatedReview):
			blankEnd = i + length/2 + 1 - len(concatenatedReview)
			for j in range(blankEnd):
				partialResult.append(0) # 0 is the empty word ""
		result.append(partialResult)
	return result



def calculateGaussian(x,mu,sigma):
	y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
	return y

def generateWeightVector(context):
	result = []
	g0 = calculateGaussian(0,0,0.1)

	for i in range(1,context+3):
		print(float(i)/(context+3)*0.3)
		g = calculateGaussian(float(i)/(context+3)*0.3,0,0.1)
		w = g/g0
		result.append(w)
	return list(reversed(result[1:-1])) + [1] + result[1:-1]

def generateInvWeightVector(context):
	w = generateWeightVector(context)
	return [1 - a for a in w]
def generateCurrWordMatrix(w,context):
	result = []
	for i in range(context*2+1):
		result.append(w)
	return result

"""
	Apply a normal distribution weighted average to the input window to reinforce the word in the middle
"""

def generateWeightedInput(x,w,context):
	wordMat = generateCurrWordMatrix(w,context)
	weightVec = generateWeightVector(context)

	m1 = []
	
	i = 0
	for w in x:
		result = [e * (1 - weightVec[i]) for e in w]
		m1.append(result)
		i+=1

	m2 = []

	i = 0
	for w in wordMat:
		result = [e * weightVec[i] for e in w]
		m2.append(result)
		i+=1

	return np.add(m1,m2)



if __name__ == '__main__':
	print(generateWeightVector(2))
	print(generateInvWeightVector(2))





