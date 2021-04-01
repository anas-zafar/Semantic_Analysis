from nltk.tokenize import word_tokenize
from collections import Counter
import string
import emojis

def EmojiCount(tweet):
    tweet = emojis.encode(tweet)
    return emojis.count(tweet, unique=True)

def ExclamationCount(tweet):
    return tweet.count('!')

def CapitalizedCount(tweet):
    return sum([1 for word in word_tokenize(tweet) if word.isupper()])

def SequenceCharacterCount(tweet):
    _sum  = 0
    for word in word_tokenize(tweet):
        frequencies = Counter(word)
        _sum = _sum  + sum([1 for key,value in frequencies.items() if value > 2])
    return _sum

def SequencePunctuationCount(tweet):
    _sum = 0
    for word in word_tokenize(tweet):
        _sum = _sum + sum([1 for p in string.punctuation if word.count(p) > 1])        
    return _sum
    

if __name__ == "__main__":
    str = 'FORMER Versace store clerk sue over secret black code for minority shoppersss !.. :thumb_up: :smile: :snake:'
    # print(emoji_count(str))
    # print(exclamation_count(str))
    # print(capitalized_count(str))
    # print(sequence_of_punctuation_count(str))
    # print(sequence_of_character_count(str))
    