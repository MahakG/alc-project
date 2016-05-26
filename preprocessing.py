""" Preprocessing corpus """
import xml.dom.minidom
import re

dictCategories = {'RESTAURANT#GENERAL':1,'RESTAURANT#PRICES':3,'RESTAURANT#MISCELLANEOUS':5,'FOOD#PRICES':7,'FOOD#QUALITY':9,'FOOD#STYLE_OPTIONS':11,
'DRINKS#PRICES':13,'DRINKS#QUALITY':15,'DRINKS#STYLE_OPTIONS':17,'AMBIENCE#GENERAL':19,'SERVICE#GENERAL':21,
'LOCATION#GENERAL':23}

############
# CLEANING #
############

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()\-,!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"- "," - ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().split()

def getText(nodelist): 
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def load_data(path):

	dom = getXmlDom(path)
	
	return handleReviews(dom)

def getXmlDom(path):
	f = open(path,'r');
	doc = ''.join(f.readlines())
	dom = xml.dom.minidom.parseString(doc)
	return dom

def handleReviews(dom):
	result = []
	reviews = dom.getElementsByTagName("Review")
	for r in reviews:
		result.append(handleReview(r))
	return result

def handleReview(r):
	result = []
	sentences = r.getElementsByTagName("sentence")
	for s in sentences:
		result.append(handleSentence(s))
	return result
def handleSentence(s):
	text = clean_str(handleText(s.getElementsByTagName("text")[0]))
	text = " ".join(text)
	opinions = handleOpinions(s.getElementsByTagName("Opinion"))
	return tagIOB(text,opinions)

def handleText(text):
	return getText(text.childNodes)

def handleOpinions(opinions):
	textAttributes = ["target","category","polarity","from","to"]
	result = []
	for o in opinions:
		attributes = []
		for attr in textAttributes:
			attributes.append(o.getAttribute(attr))
		result.append(attributes)
	return result

def tagIOB(text,opinions):
    #print(text)
    textList = [[word, 0] for word in text.split()]
    
    for o in opinions:
        targetList = o[0].split()
        if targetList[0] != 'NULL':
            pos = 0
            start = findFirstTarget(textList,targetList)

            #Check whether targetList is in textList
            while start+pos < len(textList) and pos < len(targetList) and  textList[start+pos][0] == targetList[pos]:
                pos+=1
            #If so, tag textList 

            if pos == len(targetList):
            	pos = 0
            	while pos < len(targetList):
            		
            		if pos == 0:
            			#without categories
            			textList[start+pos][1] = 1
            			#textList[start+pos][1] = dictCategories[o[1]]
            		else:
            			#without categories
            			textList[start+pos][1] = 2
            		    #textList[start+pos][1] = dictCategories[o[1]]+1

            		pos+=1
    return textList

                

def findFirstTarget(textList,targetList):
    i = 0
    while(i < len(textList) and textList[i][0] != targetList[0]):
        #print(textList[i][0],targetList[0])
        i+=1
    if i < len(textList):
        return i;
    else:
        #print(textList)
        #print(targetList)
        return -1

def getIdCategory(category):
    return dictCategories[category]

if __name__ == '__main__':
	load_data("./data/train.xml")








