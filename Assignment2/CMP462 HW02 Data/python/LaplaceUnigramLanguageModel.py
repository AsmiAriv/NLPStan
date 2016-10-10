import math, collections

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = []    
    self.wordProbs = collections.Counter()
    self.train(corpus)
    
  def wordbag(self, corpus):
    for sentence in corpus.corpus: # iterate over sentences in the corpus
      for datum in sentence.data: # iterate over datums in the sentence
        word = datum.word # get the word
        self.words.append(word)
    return self.words
    
      
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    words = self.wordbag(corpus)
    uniqueWords = collections.Counter(words)
    
    for i in set(words): # iterate over each unique word in the corpus
       probability = (uniqueWords[i]+1)/(len(words)+len(uniqueWords)) 
       self.wordProbs.update({i:probability})
    minP = 1/(len(words)+len(uniqueWords))
    self.wordProbs.update({'minP':minP})    
    return self.wordProbs    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    
    for token in sentence: # iterate over words in the sentence
      probability = self.wordProbs[token]
      if probability == 0.0:
         probability = self.wordProbs['minP']
      logProb = math.log(probability)      
      score += logProb
    return score
