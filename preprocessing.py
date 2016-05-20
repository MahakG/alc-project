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
	doc = "".join(f.readlines())
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
	text = handleText(s.getElementsByTagName("text")[0])
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
	print(text,opinions)

if __name__ == '__main__':
	load_data("./data/train.xml")