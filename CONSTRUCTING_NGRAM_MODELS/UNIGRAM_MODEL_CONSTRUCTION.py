
import nltk
import pickle

model = dict()
file = open( 'TRAIN_CORPUS.txt' , 'r' )
CORPUS = file.read()
file.close()
WORDS = nltk.tokenize.word_tokenize(CORPUS)
N = len( WORDS )
V = len( set(WORDS) )
S = len( nltk.tokenize.sent_tokenize(CORPUS) )
chars = [ '.' , '!' , '?' , '"' , '-' , ':' , ',' , "'" , ';' , '(' , ')' , '[' , ']' , '{' , '}' ]

def getEffectiveUnigramCount ( w ):
    co = WORDS.count(w)
    if not w:
        co = S
    ca = N * ( co + 0.1 ) / ( N + 0.1*V )
    return ca

types = set(nltk.tokenize.word_tokenize(CORPUS))
for w in types:
    model[w] = getEffectiveUnigramCount(w) / N

file = open( 'UNIGRAM_MODEL_PROB' , 'wb' )
pickle.dump( model , file )
file.close()
