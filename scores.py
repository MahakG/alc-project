"""
    Scores
"""
def printScores(ok,predicted,total) :

    #print("Correct: ", np.sum(ok))
    #print("Total: ", np.sum(total))
    #print("Total Predicted: ", np.sum(predicted))

    #print("Accuracy: ",calculateAccuracy(ok,predicted))

    precision = calculatePrecision(ok,predicted)
    print("Presicion(POS): ",precision[0])
    print("Presicion(NEU): ",precision[1])
    print("Presicion(NEG): ",precision[2])

    recall = calculateRecall(ok,total)
    print("Recall(POS): ",recall[0])
    print("Recall(NEU): ",recall[1])
    print("Recall(NEG): ",recall[2])

    f1 = calculateF1(precision,recall)
    print("F1(POS): ",f1[0])
    print("F1(NEU): ",f1[1])
    print("F1(NEG): ",f1[2])

    #score = (f1[0] + f1[2]) / 2.0
    #print("F1:  ",score)

def calculateAccuracy(ok,predicted):
    return np.sum(ok)/float(np.sum(predicted))

def calculatePrecision(ok,predicted):
    precision = [0,0,0]
    for i in range(len(ok)):
        if predicted[i] == 0: continue
        precision[i] = ok[i] /float(predicted[i])
    return precision
    
def calculateRecall(ok,total):
    recall = [0,0,0]
    for i in range(len(ok)):
        if total[i] == 0: continue
        recall[i] = ok[i] / float(total[i])
        
    return recall

def calculateF1(precision,recall):
    f1 = [0,0,0]
    for i in range(len(precision)):
        if precision[i]+ recall[i] == 0: continue
        f1[i] = 2 * precision[i] * recall[i] / float(precision[i] + recall[i])
    return f1

def iobF1(predictions,correct_predictions):

    detected = generateIndexes(predictions)
    correct = generateIndexes(correct_predictions)
    ok = list(set(detected).intersection(correct))
    #print(len(detected))
    #print(len(correct))
    #print(len(ok))
    if len(detected) > 0:
        precision = len(ok) / float(len(detected))
    else:
        precision = 0
    
    recall = len(ok) / float(len(correct))
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    #print("F1: ", f1)
    return [precision,recall,f1]
def generateIndexes(predictions):

    result = []
    
    begin = -1
    end = -1
    
    for i in range(len(predictions)):

        if predictions[i] == 1:
            if begin != - 1:
                result.append((begin,end))
            begin = i
            end = i
        elif predictions[i] == 2:
            end = i
        else:
            if begin != - 1:
                result.append((begin,end))
                begin = -1
                end = -1

    if begin != -1:
        result.append((begin,end))
    return result

if __name__ == '__main__':
    print(generateIndexes([0,0,1,1,2,1,0]))