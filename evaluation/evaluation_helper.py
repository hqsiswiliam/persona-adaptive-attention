#!/usr/bin/env python3
# coding:utf-8

# This source code is licensed under the MIT license.
"""
Reference Link: https://github.com/songhaoyu/BoB/blob/b369dce573a342584e594cf86c90fe34a5e7b293/evaluations.py
Reference Link: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
"""
from __future__ import division
import string
import re
from nltk.translate.bleu_score import corpus_bleu


def count_ngram(hyps_resp, n):
    """
    # Count the number of unique n-grams
    # :param hyps_resp: list, a list of responses
    # :param n: int, n-gram
    # :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


# hyps_resp = ["I am good", "you are nice",...]
def eval_distinct_avg(hyps_resp):
    """
    # compute distinct score for the hyps_resp
    # :param hyps_resp: list, a list of hyps responses
    # :return: average distinct score for 1, 2-gram
    """
    candidates = []
    for sentence in hyps_resp:
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, " {} ".format(punctuation))
        sentence = re.sub(" +", " ", sentence).strip()
        candidates.append(sentence.split(" "))
    num_tokens = sum([len(i) for i in candidates])
    dist1 = count_ngram(candidates, 1) / float(num_tokens)
    dist2 = count_ngram(candidates, 2) / float(num_tokens)

    return dist1, dist2, (dist1 + dist2) / 2.0


def eval_distinct(corpus):
    unigrams = []
    bigrams = []
    for n, rep in enumerate(corpus):
        rep = rep.strip()
        temp = rep.split(' ')
        unigrams += temp
        for i in range(len(temp) - 1):
            bigrams.append(temp[i] + ' ' + temp[i + 1])
    distink_1 = len(set(unigrams)) * 1.0 / len(unigrams)
    distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)
    return distink_1, distink_2


def write_to_file(file_path, ref_resp, hyps_resp, persona_query):
    assert len(ref_resp) == len(hyps_resp), 'length of reference is not equal to hypothesis'
    ref_resp_flatten = [i[0] for i in ref_resp]
    with open(file_path, 'w') as file:
        for query, ref, hypo in zip(persona_query, ref_resp_flatten, hyps_resp):
            file.write("QUERY:{}\n".format(query))
            file.write("REF:  {}\n".format(ref))
            file.write("HYPO: {}\n".format(hypo))
            file.write("\n")
