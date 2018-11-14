# import nltk
# nltk.download()
'''
    ▣ Installation
     - punkt
     - averaged_perceptron_tagger
'''

'''
    ▣ Tags
    ----------------------------------------------------------------------
    Number      Tag     Description
    ----------------------------------------------------------------------
    1           CC      Coordinating conjunction
    2           CD      Cardinal number
    3           DT      Determiner
    4           EX      Existential there
    5           FW      Foreign word
    6           IN      Preposition or subordinating conjunction
    7           JJ      Adjective
    8           JJR     Adjective, comparative
    9           JJS     Adjective, superlative
    10          LS      List item marker
    11          MD      Modal
    12          NN      Noun, singular or mass
    13          NNS     Noun, plural
    14          NNP     Proper noun, singular
    15          NNPS    Proper noun, plural
    16          PDT     Predeterminer
    17          POS     Possessive ending
    18          PRP     Personal pronoun
    19          PRP$    Possessive pronoun
    20          RB      Adverb
    21          RBR     Adverb, comparative
    22          RBS     Adverb, superlative
    23          RP      Particle
    24          SYM     Symbol
    25          TO      to
    26          UH      Interjection
    27          VB      Verb, base form
    28          VBD     Verb, past tense
    29          VBG     Verb, gerund or present participle
    30          VBN     Verb, past participle
    31          VBP     Verb, non-3rd person singular present
    32          VBZ     Verb, 3rd person singular present
    33          WDT     Wh-determiner
    34          WP      Wh-pronoun
    35          WP$     Possessive wh-pronoun
    36          WRB     Wh-adverb
'''

from nltk import word_tokenize, sent_tokenize, pos_tag
from collections import defaultdict
import operator
import os
import time

def reviewWordCounting(filename):
    '''
        Positive / Negative Review 에서 사용된 단어를 카운팅하는 함수
    :param filename: Review Data File
    :return: 국가별 / (Positive / Negative)별 단어 빈도수를 가지는 Dictionary 객체
    '''
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    wordDict = defaultdict(dict)  # {'nationality': {'negative': {'process': 1, 'form': 2}....

    with open(filename) as f:
        f.readline()  # 첫 째줄 건너띄기 (컬럼 값 제외)

        for line in f:
            splitLine = line.split(',')

            reviewerNationality = splitLine[5]
            negativeReview = splitLine[6]
            positiveReview = splitLine[9]

            # negative review
            if 'no negative' not in negativeReview.lower():
                negativeTags = pos_tag(word_tokenize(negativeReview))
                for word in negativeTags:
                    if word[1] in nouns:
                        # 국가별 negative key 존재 유무 확인
                        if 'negative' not in wordDict[reviewerNationality]:
                            wordDict[reviewerNationality]['negative'] = dict()

                        # 국가별 negative 별 nationality key 존재 유무 확인
                        if word[0] not in wordDict[reviewerNationality]['negative']:
                            wordDict[reviewerNationality]['negative'][word[0]] = 1
                        else:
                            wordDict[reviewerNationality]['negative'][word[0]] += 1

            # positive review
            if 'no positive' not in positiveReview.lower():
                positiveTags = pos_tag(word_tokenize(positiveReview))
                for word in positiveTags:
                    if word[1] in nouns:
                        # 국가별 positive key 존재 유무 확인
                        if 'positive' not in wordDict[reviewerNationality]:
                            wordDict[reviewerNationality]['positive'] = dict()

                        # 국가별 positive 별 nationality key 존재 유무 확인
                        if word[0] not in wordDict[reviewerNationality]['positive']:
                            wordDict[reviewerNationality]['positive'][word[0]] = 1
                        else:
                            wordDict[reviewerNationality]['positive'][word[0]] += 1

    return wordDict

def wordRankingNation(word):
    '''
        단어 빈도수를 가지는 Dictionary 객체에서 상위 TOP 100 만 추출하는 함수 (국가별)
    :param word: 단어 빈도수를 가지는 Dictionary
    :return: Positive / Negative 별 TOP 100 Dictionary 객체
    '''
    negWordRank = defaultdict(list)
    posWordRank = defaultdict(list)

    for nationKey in word.keys():
        negWordRank[nationKey] = sorted(word[nationKey]['negative'].items(), key=operator.itemgetter(1), reverse=True)[:100]
        posWordRank[nationKey] = sorted(word[nationKey]['positive'].items(), key=operator.itemgetter(1), reverse=True)[:100]

    return negWordRank, posWordRank

def wordRankingTotal(word):
    '''
        단어 빈도수를 가지는 Dictionary 객체에서 상위 TOP 100 만 추출하는 함수 (전체 국가)
    :param word: 단어 빈도수를 가지는 Dictionary
    :return: Positive / Negative 별 TOP 100 Dictionary 객체
    '''
    negWordRank = {}
    posWordRank = {}

    for nationKey in word.keys():
        for wordKey in word[nationKey]['negative'].keys():
            if wordKey not in negWordRank:
                negWordRank[wordKey] = word[nationKey]['negative'][wordKey]
            else:
                negWordRank[wordKey] += word[nationKey]['negative'][wordKey]

        for wordKey in word[nationKey]['positive'].keys():
            if wordKey not in posWordRank:
                posWordRank[wordKey] = word[nationKey]['positive'][wordKey]
            else:
                posWordRank[wordKey] += word[nationKey]['positive'][wordKey]

    negWordRank = sorted(negWordRank.items(), key=operator.itemgetter(1), reverse=True)[:100]
    posWordRank = sorted(posWordRank.items(), key=operator.itemgetter(1), reverse=True)[:100]

    return negWordRank, posWordRank

def dictToFile(negWordRankNation, posWordRankNation, negWordRankTotal, posWordRanktotal):
    rootPath = '/home/kyh/PycharmProjects/PythonRepository/Projects/AnalysisProject/result'

    '''국가별 Negative / Positive Top 100 결과 저장'''
    with open(os.path.join(rootPath, 'neg_nation_result.txt'), mode='w') as f:
        f.write("====================================================================================================\n")
        f.write("== Negative Word Ranking ===========================================================================\n")
        f.write("====================================================================================================\n")
        f.write("|         Nationality         |                           Ranking                                   \n")
        f.write("====================================================================================================\n")

        for nationKey in sorted(negWordRankNation.keys(), key=operator.itemgetter(0), reverse=False):
            nationStr = '| ' + nationKey.center(27) + ' | '
            f.write(nationStr)

            rank = 1
            for word in negWordRankNation[nationKey]:
                f.write(str(rank) + '.(' + word[0] + ', ' + str(word[1]) + ') ')
                rank += 1

            f.write('\n')

        f.write("====================================================================================================")

    with open(os.path.join(rootPath, 'pos_nation_result.txt'), mode='w') as f:
        f.write("====================================================================================================\n")
        f.write("== Positive Word Ranking ===========================================================================\n")
        f.write("====================================================================================================\n")
        f.write("|         Nationality         |                           Ranking                                   \n")
        f.write("====================================================================================================\n")

        for nationKey in sorted(posWordRankNation.keys(), key=operator.itemgetter(0), reverse=False):
            nationStr = '| ' + nationKey.center(27) + ' | '
            f.write(nationStr)

            rank = 1
            for word in posWordRankNation[nationKey]:
                f.write(str(rank) + '.(' + word[0] + ', ' + str(word[1]) + ') ')
                rank += 1

            f.write('\n')

        f.write("====================================================================================================")

    '''전체 Negative / Positive Top 100 결과 저장'''
    with open(os.path.join(rootPath, 'neg_total_result.txt'), mode='w') as f:
        f.write("====================================================================================================\n")
        f.write("== Negative Word Ranking ===========================================================================\n")
        f.write("====================================================================================================\n")
        f.write("|         Word         |     Count     |\n")
        f.write("====================================================================================================\n")

        for wordInfo in negWordRankTotal:
            wordStr = '| ' + wordInfo[0].center(20) + ' | ' + str(wordInfo[1]).center(13) + ' |\n'
            f.write(wordStr)

        f.write("====================================================================================================")

    with open(os.path.join(rootPath, 'pos_total_result.txt'), mode='w') as f:
        f.write("====================================================================================================\n")
        f.write("== Positive Word Ranking ===========================================================================\n")
        f.write("====================================================================================================\n")
        f.write("|         Word         |     Count     |\n")
        f.write("====================================================================================================\n")

        for wordInfo in posWordRanktotal:
            wordStr = '| ' + wordInfo[0].center(20) + ' | ' + str(wordInfo[1]).center(13) + ' |\n'
            f.write(wordStr)

        f.write("====================================================================================================")

startTime = time.time()

print('>> Start Review Word Counting')
reviewWord = reviewWordCounting('/home/kyh/dataset/edit_reviews.csv')

print('>> Start Word Ranking')
negWordRankNation, posWordRankNation = wordRankingNation(reviewWord)
negWordRankTotal, posWordRanktotal = wordRankingTotal(reviewWord)

print('>> Start Dict to File')
dictToFile(negWordRankNation, posWordRankNation, negWordRankTotal, posWordRanktotal)

endTime = time.time()
print('->> Elapsed Time: ', (endTime - startTime))