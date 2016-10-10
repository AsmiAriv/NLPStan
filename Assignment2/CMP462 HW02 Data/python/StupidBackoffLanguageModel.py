import math, collections


class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = []    
    self.uniquePairs = collections.Counter()    
    self.wordProbs = collections.Counter()
    self.train(corpus)
    
 
  def wordbag(self, corpus):
    for sentence in corpus.corpus: # iterate over sentences in the corpus
      for datum in sentence.data: # iterate over datums in the sentence
        word = datum.word # get the word
        self.words.append(word)
    return self.words
    
    
  def bibag(self,words):
    biword = {}
    bword = []
    for i in range(len(words)-1):
         biword[i] = words[i], words[i+1]
    bword = list(biword.values())
    self.uniquePairs = collections.Counter(bword)
    return self.uniquePairs 
     
     
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """ 
    words = self.wordbag(corpus)
    uniqueWords = collections.Counter(words)
    uniquePairs = self.bibag(words)
    list_pairs = list(uniquePairs.keys())
    
    for pair in list_pairs: # iterate over unique pair of words in the corpus
      val = uniquePairs[pair]
      
      probability = (val)/(uniqueWords[pair[0]])
      self.wordProbs.update({pair:probability})
    
    for item in set(words): # iterate over each unique word in the corpus
      probability = 0.4*(uniqueWords[item]+1)/(len(words)+len(uniqueWords))      
      self.wordProbs.update({item:probability})
      
    minP = 0.4/(len(words)+len(uniqueWords))
    self.wordProbs.update({'minP':minP})
    return self.wordProbs  
 

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    
    for i in range(len(sentence)-1): # iterate over pair of words in the sentence
      pair = sentence[i],sentence[i+1]
      probability = self.wordProbs[pair]
      if probability == 0:      
         probability = self.wordProbs[pair[1]]
         if probability == 0:
            probability = self.wordProbs['minP']
            
      logProb = math.log(probability)      
      score += logProb
    return score

