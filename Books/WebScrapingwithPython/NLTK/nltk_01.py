from nltk.book import *
from nltk import word_tokenize, sent_tokenize, pos_tag

# text = word_tokenize('''Strange women lying in ponds distributing swords is no basis for a system of government.
#                         Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony.''')
# words = pos_tag(text)
# print(words)

sentences = sent_tokenize("Google is one of the best_1st companies in the world. I constantly google myself to see what I'm up to.")
nouns = ['NN', 'NNS', 'NNP', 'NNPS']
for sentence in sentences:
    if 'google' in sentence.lower():
        taggedWords = pos_tag(word_tokenize(sentence))

        for word in taggedWords:
            if word[0].lower() == 'google' and word[1] in nouns:
                print(sentence)