
import nltk
import pickle
import math
import numpy as np

print( '\n\t[ LOADING TRAINED MODELS ... ]\n' )
file = open( 'UNIGRAM_MODEL_PROB' , 'rb' )
UNIMODEL_PR = pickle.load(file)
file.close()

file = open( 'BIGRAM_MODEL_PROB' , 'rb' )
BIMODEL_PR = pickle.load(file)
file.close()

file = open( 'BIGRAM_MODEL_COUNT' , 'rb' )
BIMODEL_C = pickle.load(file)
file.close()

file = open( 'TRIGRAM_MODEL_PROB_O1' , 'rb' )
TRIMODEL_PR = pickle.load(file)
file.close()

file = open( 'TRIGRAM_MODEL_PROB_O2' , 'rb' )
TRIMODEL_PR.update(pickle.load(file))
file.close()

file = open( 'TRIGRAM_MODEL_COUNT_O1' , 'rb' )
TRIMODEL_C = pickle.load(file)
file.close()

file = open( 'TRIGRAM_MODEL_COUNT_O2' , 'rb' )
TRIMODEL_C.update(pickle.load(file))
file.close()

file = open( 'QUADGRAM_MODEL_PROB_O1' , 'rb' )
QUADMODEL_PR = pickle.load(file)
file.close()

file = open( 'QUADGRAM_MODEL_PROB_O2' , 'rb' )
QUADMODEL_PR.update(pickle.load(file))
file.close()

file = open( 'QUADGRAM_MODEL_COUNT_O1' , 'rb' )
QUADMODEL_C = pickle.load(file)
file.close()

file = open( 'QUADGRAM_MODEL_COUNT_O2' , 'rb' )
QUADMODEL_C.update(pickle.load(file))
file.close()

file = open( 'TRAIN_CORPUS.txt' , 'r' )
CORPUS = file.read()
file.close()

file = open( 'TEST_CORPUS.txt' , 'r' )
TEST = file.read()
file.close()

UNIFORM_MODEL_PR = 1 / len( set(nltk.tokenize.word_tokenize(TEST)) )
WORDS_TRAINING = nltk.tokenize.word_tokenize(CORPUS)
SENTENCES_TESTING = nltk.tokenize.sent_tokenize(TEST)
LEN = len(nltk.tokenize.word_tokenize(TEST))
# N , V , S are characteristic to the training corpus
N = 1020640
V = 43251
S = 46235
logN = math.log10(N)

LEARNING_RATE = 0.1
ITERATIONS = 20

def getEffectiveUnigramCount ( w ):
    co = WORDS_TRAINING.count(w)
    if not w:
        co = S
    ca = N * ( co + 0.1 ) / ( N + 0.1*V )
    return ca

def getQuadgramProb ( a , b , c , d ):
    if (a,b,c) in QUADMODEL_PR:
        if d in QUADMODEL_PR[a,b,c].keys():
            return QUADMODEL_PR[a,b,c][d]
        total_count = sum(QUADMODEL_C[a,b,c].values())
        return ( 1*getEffectiveUnigramCount(d) / N ) / ( total_count + 1 )
    return 1*getEffectiveUnigramCount(d) / N   

def getTrigramProb ( a , b , c ):
    if (a,b) in TRIMODEL_PR:
        if c in TRIMODEL_PR[a,b].keys():
            return (TRIMODEL_PR[a,b][c])
        total_count = float(sum(TRIMODEL_C[a,b].values()))
        return ( 1*getEffectiveUnigramCount(c) / N ) / ( total_count + 1 )
    return 1*getEffectiveUnigramCount(c) / N

    
def getBigramProb ( a , b ):
    if a in BIMODEL_PR:
        if b in BIMODEL_PR[a].keys():
            return (BIMODEL_PR[a][b])
        total_count = float(sum(BIMODEL_C[a].values()))
        return ( 1*getEffectiveUnigramCount(b) / N ) / ( total_count + 1 )
    return 1*getEffectiveUnigramCount(b) / N

    
def getUnigramProb ( a ):
    if a in UNIMODEL_PR:
        return (UNIMODEL_PR[a])
    return 0.1 / ( N + 0.1*V )

    
def getUniformProb ( ):
    return UNIFORM_MODEL_PR


def evaluateModel ( weights ):
    l = 0
    log_perp = 0.0
    for s in SENTENCES_TESTING:
        tokens = nltk.tokenize.word_tokenize(s)
        l += len(tokens)
        bi_words = [None] + tokens + [None]
        tri_words = [None,None] + tokens + [None,None]
        quad_words = [None,None,None] + tokens + [None,None,None]
        
        for i in range(len(tokens)):
            sum = 0.0
            if weights[0]:  sum += weights[0]*getUniformProb()
            if weights[1]:  sum += weights[1]*getUnigramProb(tokens[i])
            if weights[2]:  sum += weights[2]*getBigramProb(bi_words[i],bi_words[i+1])
            if weights[3]:  sum += weights[3]*getTrigramProb(tri_words[i],tri_words[i+1],tri_words[i+2])
            if weights[4]:  sum += weights[4]*getQuadgramProb(quad_words[i],quad_words[i+1],quad_words[i+2],quad_words[i+3])
            log_perp += math.log10( sum )    
        
        i = len(tokens)
        if weights[2] or weights[3] or weights[4]:
            sum = 0.0
            if weights[2]:  sum += weights[2]*getBigramProb(bi_words[i],bi_words[i+1])
            if weights[3]:  sum += weights[3]*getTrigramProb(tri_words[i],tri_words[i+1],tri_words[i+2])
            if weights[4]:  sum += weights[4]*getQuadgramProb(quad_words[i],quad_words[i+1],quad_words[i+2],quad_words[i+3])
            log_perp += math.log10( sum )
        
        i += 1
        if weights[3] or weights[4]:
            sum = 0.0
            if weights[3]:  sum += weights[3]*getTrigramProb(tri_words[i],tri_words[i+1],tri_words[i+2])
            if weights[4]:  sum += weights[4]*getQuadgramProb(quad_words[i],quad_words[i+1],quad_words[i+2],quad_words[i+3])
            log_perp += math.log10( sum )
        
        i += 1
        if weights[4]:
            log_perp += math.log10( weights[4]*getQuadgramProb(quad_words[i],quad_words[i+1],quad_words[i+2],quad_words[i+3]) )
    
    log_perp *= ( -1 / l )
    return round(math.pow(10,log_perp))

def optimiseInterpolatedUniModel ( weights ):
    for i in range(ITERATIONS):
        grad = 0.0
        for s in SENTENCES_TESTING:
            ngram_words = nltk.tokenize.word_tokenize(s)
            for k in range(len(tokens)):
                uniform = getUniformProb()
                ngram = getUnigramProb(ngram_words[k])
                grad += (ngram-uniform) / (weights[0]*uniform+weights[1]*ngram)
         
        grad /= LEN
        weights[1] += LEARNING_RATE*grad
        weights[0] = 1 - weights[1]
        print( ' WEIGHTS AFTER' , i+1 , 'ITERATIONS :' , weights )

  
def optimiseInterpolatedBiModel ( weights ):
    print( ' INITIAL WEIGHTS :' , weights)
    for i in range(ITERATIONS):
        grad1 = 0.0
        grad2 = 0.0
        for s in SENTENCES_TESTING:
            tokens = nltk.tokenize.word_tokenize(s)
            biwords = [None] + tokens + [None]
            for k in range(len(tokens)):
                uniform = getUniformProb()
                uni = getUnigramProb(tokens[k])
                bi = getBigramProb(biwords[k], biwords[k+1])
                interpolP = weights[0]*uniform + weights[1]*uni + weights[2]*bi
                grad1 += (uni - uniform) / interpolP
                grad2 += (bi - uniform) / interpolP
            k = len(tokens)
            bi = getBigramProb(biwords[k], biwords[k+1])
            grad2 += bi / (weights[2]*bi)
            k += 1
        
        grad2 /= LEN
        grad1 /= LEN
        weights[1] += LEARNING_RATE*grad1
        weights[2] += LEARNING_RATE*grad2
        weights[0] = 1 - weights[1] - weights[2]
        print( ' WEIGHTS AFTER' , i+1 , 'ITERATIONS :' , weights )
        

def optimiseInterpolatedTriModel ( weights ):
    print( ' INITIAL WEIGHTS :' , weights)
    for i in range(ITERATIONS):
        grad1 = 0.0
        grad2 = 0.0
        grad3 = 0.0
        for s in SENTENCES_TESTING:
            tokens = nltk.tokenize.word_tokenize(s)
            biwords = [None] + tokens + [None]
            triwords = [None] + biwords + [None]
            for k in range(len(tokens)):
                uniform = getUniformProb()
                uni = getUnigramProb(tokens[k])
                bi = getBigramProb(biwords[k], biwords[k+1])
                tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
                interpolP = weights[0]*uniform + weights[1]*uni + weights[2]*bi + weights[3]*tri
                grad1 += (uni - uniform) / interpolP
                grad2 += (bi - uniform) / interpolP
                grad3 += (tri - uniform) / interpolP
            
            k = len(tokens)
            bi = getBigramProb(biwords[k], biwords[k+1])
            tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
            interpolP = weights[2]*bi + weights[3]*tri
            grad2 += bi / interpolP
            grad3 += tri / interpolP   
                
            k += 1
            tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
            grad3 += tri / (weights[3]*tri) 
        
        grad3 /= LEN
        grad2 /= LEN
        grad1 /= LEN
        weights[1] += LEARNING_RATE*grad1
        weights[2] += LEARNING_RATE*grad2
        weights[3] += LEARNING_RATE*grad3
        weights[0] = 1 - weights[1] - weights[2] - weights[3]
        print( ' WEIGHTS AFTER' , i+1 , 'ITERATIONS :' , weights )


def optimiseInterpolatedQuadModel ( weights ):
    print( ' INITIAL WEIGHTS :' , weights)
    for i in range(ITERATIONS):
        grad1 = 0.0
        grad2 = 0.0
        grad3 = 0.0
        grad4 = 0.0
        for s in SENTENCES_TESTING:
            tokens = nltk.tokenize.word_tokenize(s)
            biwords = [None] + tokens + [None]
            triwords = [None] + biwords + [None]
            quadwords = [None] + triwords + [None]
            for k in range(len(tokens)):
                uniform = getUniformProb()
                uni = getUnigramProb(tokens[k])
                bi = getBigramProb(biwords[k], biwords[k+1])
                tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
                quad = getQuadgramProb(quadwords[k], quadwords[k+1], quadwords[k+2], quadwords[k+3])
                interpolP = weights[0]*uniform + weights[1]*uni + weights[2]*bi + weights[3]*tri + weights[4]*quad
                grad1 += (uni-uniform) / interpolP
                grad2 += (bi-uniform) / interpolP
                grad3 += (tri-uniform) / interpolP
                grad4 += (quad-uniform) / interpolP
            
            k = len(tokens)
            bi = getBigramProb(biwords[k], biwords[k+1])
            tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
            quad = getQuadgramProb(quadwords[k], quadwords[k+1], quadwords[k+2], quadwords[k+3])
            interpolP = weights[2]*bi + weights[3]*tri + weights[4]*quad
            grad2 += bi / interpolP
            grad3 += tri / interpolP
            grad4 += quad / interpolP
            k += 1
        
            tri = getTrigramProb(triwords[k], triwords[k+1], triwords[k+2])
            quad = getQuadgramProb(quadwords[k], quadwords[k+1], quadwords[k+2], quadwords[k+3])
            interpolP = weights[3]*tri + weights[4]*quad
            grad3 += tri / interpolP
            grad4 += quad / interpolP
            k += 1
            
            quad = getQuadgramProb(quadwords[k], quadwords[k+1], quadwords[k+2], quadwords[k+3])
            grad4 += quad / (weights[4]*quad)
        
        grad4 /= LEN
        grad3 /= LEN
        grad2 /= LEN
        grad1 /= LEN
        weights[1] += LEARNING_RATE*grad1
        weights[2] += LEARNING_RATE*grad2
        weights[3] += LEARNING_RATE*grad3
        weights[4] += LEARNING_RATE*grad4
        weights[0] = 1 - weights[1] - weights[2] - weights[3] - weights[4]
        print( ' WEIGHTS AFTER' , i+1 , 'ITERATIONS :' , weights )

# before running the program, set the value of n
# to indicate the interpolated model whose
# performance has to be optimized 
n = 3
# before running the program, set an initial distribution 
# of weights, WHILE KEEPING THEIR SUM TOTAL AS 1. The weights
# corresponding to the models with gram-length more than n
# MUST BE ZERO.
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.0])

if n == 1:    optimiseInterpolatedUniModel(weights)
elif n == 2:  optimiseInterpolatedBiModel(weights)
elif n == 3:  optimiseInterpolatedTriModel(weights)
elif n == 4:  optimiseInterpolatedQuadModel(weights)
print( '\n  OPTIMISED PERPLEXITY :' , evaluateModel(weights) , '\n\n' )
