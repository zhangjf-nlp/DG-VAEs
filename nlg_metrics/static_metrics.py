from .utils.bleu_scorer import BleuScorer
from .utils.insert_spaces import insert_spaces
from .rouge.rouge import Rouge
import numpy as np
from tqdm import tqdm

class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n

    def compute_score(self, gts, res):
        gts = insert_spaces(gts)
        res = insert_spaces(res)
        
        assert(gts.keys() == res.keys())

        bleu_scorer = BleuScorer(n=self._n)
        for intent in gts:
            hypo = res[intent]
            ref = gts[intent]
            for hyp in hypo:
                bleu_scorer += (hyp, ref)
                
        #score, scores = bleu_scorer.compute_score(option='shortest')
        #score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        score, scores = bleu_scorer.compute_score(option='average', verbose=0)
        
        return score

    def method(self):
        return "Bleu"

Bleu = Bleu()

def bleu(gts, res):
    """ compute the bleu-1/2/3/4 scores between the input strings
    Args:
        res (dict[obj:list[str]]): the input list of string or a dict of string list
        gts (dict[obj:list[str]]): the same as res, and it must have the same keys to that of res
    """
    return Bleu.compute_score(gts, res)



def word_cleaner(x):
    return x.strip('\r\n')

def _clean(sentence):
    sentence = word_cleaner(sentence)
    return sentence

def _distinct(max_order, fh):
    translations = []
    
    for line in fh:
        line = _clean(line)
        translations.append(line.split(" "))  

    num_tokens = 0
    unique_tokens = set()
    scores = []
    for items in translations:
        local_unique_tokens = set()
        local_count = 0.0
        #print(items) 
        for i in range(0, len(items) - max_order + 1):
            tmp = ' '.join(items[i:i+max_order])

            unique_tokens.add(tmp)
            num_tokens += 1
            local_unique_tokens.add(tmp)
            local_count += 1

        if local_count == 0:
            scores.append(0)
        else:
            scores.append(100*len(local_unique_tokens) / local_count)
            
  
    if num_tokens == 0:
        ratio = 0
    else:
        ratio = len(unique_tokens) / num_tokens
    
    """
    if max_order == 1:
        print('distinct1: %s\n\t' % (100*ratio) )
    elif max_order == 2:
        print('distinct2: %s\n\t' % (100*ratio) )
    #print('distinct_scores: %s\n\n' % ( scores))"""

    return 100 * ratio

# add tokenization before computing the score
def distinct(res):
    """ compute the distinct-1/2/3/4 scores between the input strings
    Args:
        res (list[str] or dict[obj:list[str]]): the input list of string or a dict of string list
    """
    if type(res) is list:
        return [_distinct(max_order, insert_spaces(res)) for max_order in [1,2,3,4]]
    elif type(res) is dict:
        scores = np.array([distinct(res[intent]) for intent in tqdm(res)])
        weight = np.array([len(res[intent]) for intent in res])
        weight = weight / np.sum(weight)
        score = np.sum(weight[:,None] * scores, axis=0)
        return score
    else:
        assert False, f"unsupported input type: {type(input)}"



class SelfBleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        
    def compute_score(self, res):
        if type(res) is list:
            res = insert_spaces(res)
            bleu_scorer = BleuScorer(n=self._n)
            for i,state in enumerate(res):
                bleu_scorer += (state, res[:i]+res[i+1:])
            score, _ = bleu_scorer.compute_score(option='average', verbose=0)
            return score
        elif type(res) is dict:
            scores = np.array([self.compute_score(res[intent]) for intent in tqdm(res)])
            # weighted average
            weight = np.array([len(res[intent]) for intent in res])
            weight = weight / np.sum(weight)
            score = np.sum(weight[:,None] * scores, axis=0)
            return score
        else:
            assert False, f"unsupported input type: {type(input)}"
            
    def method(self):
        return "Self-Bleu"

SelfBleu = SelfBleu()

def selfbleu(input):
    """ mean([bleu(a,b) for a in input for b in input-[a]])  https://www.aclweb.org/anthology/P19-1177.pdf
    Args:
        input (list[str] or dict[obj:list[str]]): the input list of string or a dict of string list
    """
    return SelfBleu.compute_score(input)

Rouge = Rouge()

def rouge(gts, res, avg=True):
    """
    Args:
        gts, res: (list[str])
    """
    assert type(gts)==type(res)
    assert type(gts) is list
    return Rouge.get_scores(res, gts, avg)
        