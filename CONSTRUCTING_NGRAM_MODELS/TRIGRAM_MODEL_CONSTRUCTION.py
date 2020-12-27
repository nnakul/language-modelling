
import nltk
from nltk import trigrams
import pickle

model = dict()
file = open( 'TRAIN_CORPUS.txt' , 'r' )
CORPUS = file.read()
file.close()

WORDS = nltk.tokenize.word_tokenize(CORPUS)
N = len( WORDS )
V = len( set(WORDS) )
SENTENCES = nltk.tokenize.sent_tokenize(CORPUS)
S = len( SENTENCES )
chars = [ '.' , '!' , '?' , '"' , '-' , ':' , ',' , "'" , ';' , '(' , ')' , '[' , ']' , '{' , '}' ]

def getEffectiveUnigramCount ( w ):
    co = WORDS.count(w)
    if not w:
        co = S
    ca = N * ( co + 0.1 ) / ( N + 0.1*V )
    return ca
    
for s in SENTENCES:
    words = nltk.tokenize.word_tokenize(s)
    for w1, w2, w3 in trigrams(words, pad_right=True, pad_left=True):
        if not (w1,w2) in model:
            model[(w1,w2)] = dict()
        if not w3 in model[(w1, w2)]:
            model[(w1,w2)][w3] = 0
        model[(w1, w2)][w3] += 1
 
modelc = model.copy()
for w1_w2 in model:
    total_count = float(sum(modelc[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] = ( model[w1_w2][w3] + 1*getEffectiveUnigramCount(w3)/N ) / ( total_count + 1 )

file = open( 'model' , 'wb' )
pickle.dump( model , file )
file.close()

file = open( 'model_count' , 'wb' )
pickle.dump( modelc , file )
file.close()
