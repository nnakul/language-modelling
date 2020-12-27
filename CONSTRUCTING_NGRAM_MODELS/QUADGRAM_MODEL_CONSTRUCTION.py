
import nltk
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
    tokens = nltk.tokenize.word_tokenize(s)
    words = [ None , None , None ] + tokens + [ None , None , None ]
    for i in range(3,len(words)):
        if not (words[i-3],words[i-2],words[i-1]) in model:
            model[words[i-3],words[i-2],words[i-1]] = dict()
        if not words[i] in model[words[i-3],words[i-2],words[i-1]]:
            model[words[i-3],words[i-2],words[i-1]][words[i]] = 0
        model[words[i-3],words[i-2],words[i-1]][words[i]] += 1

modelc = model.copy()
for w123 in model:
    total_count = float(sum(modelc[w123].values()))
    for w4 in model[w123]:
        model[w123][w4] = ( model[w123][w4] + 1*getEffectiveUnigramCount(w4)/N ) / ( total_count + 1 )

file = open( 'QUADGRAM_MODEL_PROB' , 'wb' )
pickle.dump( model , file )
file.close()

file = open( 'QUADGRAM_MODEL_COUNT' , 'wb' )
pickle.dump( modelc , file )
file.close()
