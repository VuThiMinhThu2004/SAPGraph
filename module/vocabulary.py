#!/usr/bin/python
# -*- coding: utf-8 -*-

from tools.logger import *


PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        :param vocab_file: string; path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
        :param max_size: int; The maximum size of the resulting Vocabulary.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN,  START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
        
        # add entity into vocab first
        with open(vocab_file+'_ent', 'r', encoding='utf8') as ent_f:
            cnt = 0
            for line in ent_f:
                cnt+=1
                pieces = line.split("\t")
                w = pieces[1]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    # logger.error('Duplicated entity in entity file Line %d : %s' % (cnt, w))
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    logger.info("[INFO] max_size of entity vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        
        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f: #New : add the utf8 encoding to prevent error
            cnt = 0
            dup = 0
            for line in vocab_f:
                cnt+=1
                pieces = line.split("\t")
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    dup +=1
                    # logger.error('Duplicated word or entity in vocabulary file Line %d : %s' % (cnt, w))
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    logger.info("[INFO] max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
            logger.info("[INFO] %i of word vocab was already in entity." % (dup))
        logger.info("[INFO] Finished constructing vocabulary of %i total words. Last word added: %s", self._count, self._id_to_word[self._count-1])

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def word_list(self):
        """Return the word list of the vocabulary"""
        return self._word_to_id.keys()