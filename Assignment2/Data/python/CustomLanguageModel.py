import math, collections


class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = []
    self.uniqueWords = collections.Counter()
    self.uniquePairs = collections.Counter()    
    self.wordProbs = collections.Counter()
    self.pkn = collections.Counter()
    self.lamb = collections.Counter()
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
    self.wordbag(corpus)
    self.uniqueWords = collections.Counter(self.words)
    self.bibag(self.words)
    list_pairs = list(self.uniquePairs.keys())
    
    d = 0.5
    sump=0
    dump=0
    
    for word in set(self.words):
     for item in list_pairs:
        if item[1] == word:
            sump += 1
        if item[0] == word:
            dump += 1    
     pkn = (sump)/(len(self.uniquePairs))
     lamb = d*(dump)/(self.uniqueWords[word])
     self.pkn.update({word:pkn})
     self.lamb.update({word:lamb})
        
    for pair in list_pairs: # iterate over unique pair of words in the corpus
      val = self.uniquePairs[pair]
      pkn = self.pkn[pair[1]]
      lamb = self.lamb[pair[0]]
      term1 = (max((val-d),0)/(self.uniqueWords[pair[0]]))
      term2 = lamb*pkn
      probability = term1 + term2
      self.wordProbs.update({pair:probability})
    minP = 1/(len(self.uniqueWords)+len(self.words))
    self.wordProbs.update({'minP':minP})    
       

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    
    score = 0.0
       
    for i in range(len(sentence)-1): # iterate over pair of words in the sentence
      pair = sentence[i],sentence[i+1]
      pkn = self.pkn[pair[1]]
      probability = self.wordProbs[pair]
      
      if probability == 0:
         probability = pkn
                
      if probability == 0:
         probability = self.wordProbs['minP'] 
      
      logProb = math.log(probability)      
      score += logProb
    return score

