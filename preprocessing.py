""" Preprocessing corpus """
import xml.dom.minidom

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
dictCategories = {'RESTAURANT#GENERAL':1,'RESTAURANT#PRICES':3,'RESTAURANT#QUALITY':5,'RESTAURANT#STYLE_OPTIONS':7,'RESTAURANT#MISCELLANEOUS':9,'FOOD#GENERAL':11,'FOOD#PRICES':13,'FOOD#QUALITY':15,'FOOD#STYLE_OPTIONS':17,'FOOD#MISCELLANEOUS':19,'DRINKS#GENERAL':21,'DRINKS#PRICES':23,'DRINKS#QUALITY':25,'DRINKS#STYLE_OPTIONS':27,'DRINKS#MISCELLANEOUS':29,'AMBIENCE#GENERAL':31,'AMBIENCE#PRICES':33,'AMBIENCE#QUALITY':35,'AMBIENCE#STYLE_OPTIONS':37,'AMBIENCE#MISCELLANEOUS':39,'SERVICE#GENERAL':41,'SERVICE#PRICES':43,'SERVICE#QUALITY':45,'SERVICE#STYLE_OPTIONS':47,'SERVICE#MISCELLANEOUS':49,'LOCATION#GENERAL':51,'LOCATION#PRICES':53,'LOCATION#QUALITY':55,'LOCATION#STYLE_OPTIONS':57,'LOCATION#MISCELLANEOUS':59}
def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def load_data(path):

	dom = getXmlDom(path)
	
	handleReviews(dom)

def getXmlDom(path):
	f = open(path,'r');
	doc = ''.join(f.readlines())
	dom = xml.dom.minidom.parseString(doc)
	return dom

def handleReviews(dom):
	reviews = dom.getElementsByTagName("Review")
	for r in reviews:
		handleReview(r)

def handleReview(r):
	sentences = r.getElementsByTagName("sentence")
	for s in sentences:
		handleSentence(s)

def handleSentence(s):
	text = removePunctuation(handleText(s.getElementsByTagName("text")[0]))
	opinions = handleOpinions(s.getElementsByTagName("Opinion"))
	tagIOB(text,opinions)

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
    print(text)
    textList = [[word, 0] for word in text.split()]
    
    for o in opinions:
        targetList = o[0].split()
        if targetList[0] != 'NULL':
            print(findFirstTarget(textList,targetList))
        
def findFirstTarget(textList,targetList):
    i = 0
    while(i < len(textList) and textList[i][0] != targetList[0]):
        #print(textList[i][0],targetList[0])
        i+=1
    if i < len(textList):
        return i;
    else:
        return -1
def getIdCategory(category):
    return dictCategories[category]

if __name__ == '__main__':
	load_data("./data/train.xml")



############
# CLEANING #
############

def remove_punct(text):
  tokens = text.split();
  punctuation = """, . 
  ¡ ! 
  ? ¿ = )
   ( / & % $ · [ ] { } 
    - _ * ^ : \" &lt; &gt; 
    RT ... ' """
  return [word for word in tokens if word not in punctuation.split()]


