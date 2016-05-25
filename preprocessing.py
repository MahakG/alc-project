""" Preprocessing corpus """
import xml.dom.minidom
import re
"""
document = ""\
<slideshow>
<title>Demo slideshow</title>
<slide><title>Slide title</title>
<point>This is a demo</point>
<point>Of a program for processing slides</point>
</slide>

<slide><title>Another demo slide</title>
<point>It is important</point>
<point>To have more than</point>
<point>one slide</point>
</slide>
</slideshow>
""

dom = xml.dom.minidom.parseString(document)

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def handleSlideshow(slideshow):
    print "<html>"
    handleSlideshowTitle(slideshow.getElementsByTagName("title")[0])
    slides = slideshow.getElementsByTagName("slide")
    handleToc(slides)
    handleSlides(slides)
    print "</html>"

def handleSlides(slides):
    for slide in slides:
        handleSlide(slide)

def handleSlide(slide):
    handleSlideTitle(slide.getElementsByTagName("title")[0])
    handlePoints(slide.getElementsByTagName("point"))

def handleSlideshowTitle(title):
    print "<title>%s</title>" % getText(title.childNodes)

def handleSlideTitle(title):
    print "<h2>%s</h2>" % getText(title.childNodes)

def handlePoints(points):
    print "<ul>"
    for point in points:
        handlePoint(point)
    print "</ul>"

def handlePoint(point):
    print "<li>%s</li>" % getText(point.childNodes)

def handleToc(slides):
    for slide in slides:
        title = slide.getElementsByTagName("title")[0]
        print "<p>%s</p>" % getText(title.childNodes)

handleSlideshow(dom)
"""
dictCategories = {'RESTAURANT#GENERAL':1,'RESTAURANT#PRICES':3,'RESTAURANT#MISCELLANEOUS':5,'FOOD#PRICES':7,'FOOD#QUALITY':9,'FOOD#STYLE_OPTIONS':11,
'DRINKS#PRICES':13,'DRINKS#QUALITY':15,'DRINKS#STYLE_OPTIONS':17,'AMBIENCE#GENERAL':19,'SERVICE#GENERAL':21,
'LOCATION#GENERAL':23}

############
# CLEANING #
############

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9()\-,!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"- "," - ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def remove_punct(text):
	tokens = text.split();
	punctuation = """, . ¡ ! ? ¿ = ) ( / & % $ · [ ] { } - _ * ^ : \" &lt; &gt; RT ... ' """
	return [word for word in tokens if word not in punctuation.split()]

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def load_data(path):

	dom = getXmlDom(path)
	
	print(handleReviews(dom))

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
	text = remove_punct(clean_str(handleText(s.getElementsByTagName("text")[0])))
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
            			#textList[start+pos][1] = 1
            			textList[start+pos][1] = dictCategories[o[1]]
            		else:
            			#without categories
            			#textList[start+pos][1] = 2
            		    textList[start+pos][1] = dictCategories[o[1]]+1

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








