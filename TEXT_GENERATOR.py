
import nltk
import random
import pickle
import time

file = open( 'UNIGRAM_MODEL_PROB' , 'rb' )
UNIMODEL_PR = pickle.load(file)
file.close()

file = open( 'BIGRAM_MODEL_PROB' , 'rb' )
BIMODEL_PR = pickle.load(file)
file.close()

file = open( 'TRIGRAM_MODEL_PROB_O1' , 'rb' )
TRIMODEL_PR = pickle.load(file)
file.close()

file = open( 'TRIGRAM_MODEL_PROB_O2' , 'rb' )
TRIMODEL_PR.update(pickle.load(file))
file.close()

file = open( 'QUADGRAM_MODEL_PROB_O1' , 'rb' )
QUADMODEL_PR = pickle.load(file)
file.close()

file = open( 'QUADGRAM_MODEL_PROB_O2' , 'rb' )
QUADMODEL_PR.update(pickle.load(file))
file.close()

text = list()
chars = '.!?-:,;' + "'"
delims = '.?!"'
brackets_open = '([{"'
brackets_close = ')]}"'

def tokensToSentence( text ):
    str_txt = ''
    str_txt += text[0].capitalize()
    for i in range(1,len(text)):
        if not text[i]:
            continue
        if text[i-1] in delims:
            text[i] = text[i].capitalize()
        if text[i][0] in chars:
            str_txt += text[i]
        elif text[i][0] in brackets_open:
            str_txt += ' ' + text[i]
        elif text[i][0] in brackets_close:
            str_txt += text[i]
        elif "'" in text[i]:
            str_txt += text[i]
        elif text[i-1][0] in brackets_open:
            str_txt += text[i]
        else:
            str_txt += ' ' + text[i]
    return str_txt


def completeUsingUnigramModel ( text ):
    while True:
        thresh = random.random()
        acc = 0.0
        for w in UNIMODEL_PR:
            acc += UNIMODEL_PR[w]
            if acc >= thresh:
                text.append(w)
                break
        if text[-1:][0] in delims:
            break        
    return tokensToSentence(text)
    print( '\n' , tokensToSentence(text) )


def completeUsingBigramModel ( text ):
    text = [ None ] + text
    
    while ( not text[-1:][0] in BIMODEL_PR ):
        thresh = random.random()
        acc = 0.0
        for w in UNIMODEL_PR:
            acc += UNIMODEL_PR[w]
            if acc >= thresh:
                text.append(w)
                break
            
    while True:
        if not text[-1:][0]:
            break
        thresh = random.random()
        acc = 0.0
        prev = text[-1:][0]
        for w in BIMODEL_PR[prev]:
            acc += BIMODEL_PR[prev][w]
            if acc >= thresh:
                text.append(w)
                break
    return tokensToSentence(text[1:])
    print( '\n' , tokensToSentence(text[1:]) )
    
    
def completeUsingTrigramModel ( text ):
    text = [ None , None ] + text
    
    while ( not tuple(text[-2:]) in TRIMODEL_PR ):
        if not text[-1:][0] in BIMODEL_PR:
            thresh = random.random()
            acc = 0.0
            for w in UNIMODEL_PR:
                acc += UNIMODEL_PR[w]
                if acc >= thresh:
                    text.append(w)
                    break
        else:
            thresh = random.random()
            acc = 0.0
            prev = text[-1:][0]
            for w in BIMODEL_PR[prev]:
                acc += BIMODEL_PR[prev][w]
                if acc >= thresh:
                    text.append(w)
                    break
                    
    while True:
        if not text[-1:][0]:
            break
        thresh = random.random()
        acc = 0.0
        prev = tuple(text[-2:])
        for w in TRIMODEL_PR[prev]:
            acc += TRIMODEL_PR[prev][w]
            if acc >= thresh:
                text.append(w)
                break
    return tokensToSentence(text[2:])        
    print( '\n' , tokensToSentence(text[2:]) )


def completeUsingQuadgramModel ( text ):
    text = [None , None , None] + text
    
    while ( not tuple(text[-3:]) in QUADMODEL_PR ):
        if not tuple(text[-2:]) in TRIMODEL_PR:
            if not text[-1:][0] in BIMODEL_PR:
                thresh = random.random()
                acc = 0.0
                for w in UNIMODEL_PR:
                    acc += UNIMODEL_PR[w]
                    if acc >= thresh:
                        text.append(w)
                        break
            else:
                thresh = random.random()
                acc = 0.0
                prev = text[-1:][0]
                for w in BIMODEL_PR[prev]:
                    acc += BIMODEL_PR[prev][w]
                    if acc >= thresh:
                        text.append(w)
                        break
        else:
            thresh = random.random()
            acc = 0.0
            prev = tuple(text[-2:])
            for w in TRIMODEL_PR[prev]:
                acc += TRIMODEL_PR[prev][w]
                if acc >= thresh:
                    text.append(w)
                    break
                    
                    
    while True:
        if not text[-1:][0]:
            break
        thresh = random.random()
        acc = 0.0
        prev = tuple(text[-3:])
        for w in QUADMODEL_PR[prev]:
            acc += QUADMODEL_PR[prev][w]
            if acc >= thresh:
                text.append(w)
                break
    return tokensToSentence(text[3:])



def produceResults ( count , model , query_tokens , allowed_delay ):
    results = list()
    for i in range(count):
        text = query_tokens.copy()
        if model == 1:      
            r = completeUsingUnigramModel( text )
            ts = time.time()
            while r in results:
                if time.time() - ts > allowed_delay:
                    return
                text = query_tokens.copy()
                r = completeUsingUnigramModel( text )
            print( '\n' , r )
            results.append(r)
        elif model == 2:    
            r = completeUsingBigramModel( text )
            ts = time.time()
            while r in results:
                if time.time() - ts > allowed_delay:
                    return
                text = query_tokens.copy()
                r = completeUsingBigramModel( text )
            print( '\n' , r )
            results.append(r)
        elif model == 3:    
            r = completeUsingTrigramModel( text )
            ts = time.time()
            while r in results:
                if time.time() - ts > allowed_delay:
                    return
                text = query_tokens.copy()
                r = completeUsingTrigramModel( text )
            print( '\n' , r )
            results.append(r)
        elif model == 4:    
            r = completeUsingQuadgramModel( text )
            ts = time.time()
            while r in results:
                if time.time() - ts > allowed_delay:
                    return
                text = query_tokens.copy()
                r = completeUsingQuadgramModel( text )
            print( '\n' , r )
            results.append(r)
    
    
    
    

while True:
    print('\n')
    query = input( 'ENTER INCOMPLETE TEXT : ' )
    if query.lower() == '#quit': break
    query_tokens = nltk.tokenize.word_tokenize(query)
    if not len(query_tokens):
        print( '\n  **INVALID TEXT**' )
        continue
    for w in query_tokens:
        w = w.lower()
    model = input( 'ENTER PREFERABLE MODEL (1:UNI, 2:BI, 3:TRI, 4:QUAD) : ' )
    try:
        model = int(model)
    except:
        print( '\n  **INVALID PREFERANCE OF MODEL**' )
        continue
    if not model in {1,2,3,4}:
        print( '\n  **INVALID PREFERANCE OF MODEL**' )
        continue
    num = input( 'ENTER MAX NUMBER OF RESULTS : ' )
    try:
        num = int(num)
    except:
        print( '\n  **INVALID NUMBER OF RESULTS**' )
        continue
    if num<=0:
        print( '\n  **INVALID NUMBER OF RESULTS**' )
        continue
    
    produceResults(num,model,query_tokens,3)

print( '\n\n  CLOSING...' )
time.sleep(1)