'''
Interface for converting POS tags from various treebanks 
to the universal tagset of Petrov, Das, & McDonald.

The tagset consists of the following 12 coarse tags:

VERB - verbs (all tenses and modes)
NOUN - nouns (common and proper)
PRON - pronouns 
ADJ - adjectives
ADV - adverbs
ADP - adpositions (prepositions and postpositions)
CONJ - conjunctions
DET - determiners
NUM - cardinal numbers
PRT - particles or other function words
X - other: foreign words, typos, abbreviations
. - punctuation

@see: http://arxiv.org/abs/1104.2086 and http://code.google.com/p/universal-pos-tags/

@author: Nathan Schneider (nschneid)
@since: 2011-05-06
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division
from future_builtins import map, filter

import re, glob
from collections import defaultdict

MAP_DIR = 'universal_pos_tags.1.01'

COARSE_TAGS = ('VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.')

_MAPS = defaultdict(dict)

def readme():
    with open(MAP_DIR+'/README') as f:
        return f.read()

def fileids(lang=''):
    '''
    Optionally given a two-letter ISO language code, returns names of files 
    containing mappings from a tagset from a treebank in that language to the 
    universal tagset.
    
    >>> fileids('en')
    [u'en-ptb']
    >>> fileids('zh')
    [u'zh-ctb6', u'zh-sinica']
    ''' 
    return [re.match(r'.*[/]([^/\\]+)[.]map', p).group(1) for p in glob.glob(MAP_DIR + '/{}-*.map'.format(lang.lower()))]

def _read(fileid):
    with open(MAP_DIR+'/'+fileid+'.map') as f:
        for ln in f:
            ln = ln.strip()
            if ln=='': continue
            fine, coarse = ln.split('\t')
            assert coarse in COARSE_TAGS,'Unexpected coarse tag: {}'.format(coarse)
            assert fine not in _MAPS[fileid],'Multiple entries for original tag: {}'.format(fine)
            _MAPS[fileid][fine] = coarse
            
def mapping(fileid):
    '''
    Retrieves the mapping from original tags to universal tags for the 
    treebank in question.
    
    >>> mapping('ru-rnc')=={'!': '.', 'A': 'ADJ', 'AD': 'ADV', 'C': 'CONJ', 'COMP': 'CONJ', 'IJ': 'X', 'NC': 'NUM', 'NN': 'NOUN', 'P': 'PRON', 'PTCL': 'PRT', 'V': 'VERB', 'VG': 'VERB', 'VI': 'VERB', 'VP': 'VERB', 'YES_NO_SENT': 'X', 'Z': 'X'}
    True
    '''
    if fileid not in _MAPS:
        _read(fileid)
    return _MAPS[fileid]

def convert(fileid, originalTag):
    '''
    Produces the (coarse) universal tag given an original POS tag from the 
    treebank in question.
    
    >>> convert('en-ptb', 'VBZ')
    u'VERB'
    >>> convert('en-ptb', 'VBP')
    u'VERB'
    >>> convert('en-ptb', '``')
    u'.'
    '''
    return mapping(fileid)[originalTag]


def test():
    for fileid in fileids():
        mapping(fileid)
    import doctest
    doctest.testmod()

if __name__=='__main__':
    test()
