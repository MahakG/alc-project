
l = ['RESTAURANT','FOOD','DRINKS','AMBIENCE','SERVICE','LOCATION']
p = ['GENERAL','PRICES','QUALITY','STYLE_OPTIONS','MISCELLANEOUS']

result = ''
counter = 1
for a in l:
	for b in p:
		result += '\''+a+'#'+b+'\':'+str(counter)+","
		counter+=2
print(result)