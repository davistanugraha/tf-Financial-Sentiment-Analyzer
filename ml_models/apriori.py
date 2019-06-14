import operator
#import spacy
import logging
import pandas as pd
import nltk
#from spacy.tokenizer import Tokenizer

class Apriori_Model():
    """
    This is a class for predicting text using APRIORI algorithm

    Attributes:
        NLP (spacy object): trained spacy model.
        TAGGING(dict): dictionary containing lexicon information use for tagging
        RULES (list): list of dictionaries containing association rules for assigning sentiment
    """
    def __init__(self):
        """
        constructor for Apriori_Model class.
        """
        # loading spacy trained model for tokenizing
        #self.NLP = spacy.load("en_core_web_sm")
        # loading dictionary for tagging
        self.TAGGING = pd.read_pickle("resources/pickle/word_list_dict.pkl")
        # loading APRIORI association rules
        self.RULES = pd.read_pickle("resources/pickle/apriori_rules.pkl")
        #loading tokenizer
        #self.tokenizer = Tokenizer(self.NLP.vocab)

    def replace_words_with_tags(self, sentence, lexicon_categories):
        """
        replace words that belongs in lexicons with the lexicons title.
        :param sentence: string representing text that will be tagged
        :param lexicon_categories: tagging dictionary
        :param spacy_tokenizer: spacy loaded model
        :return: string with words that belong in the lexicon replaced by the tags.
        """
        # counter loops through each word in a sentence
        counter = 0
        # tokenize_text list of tokens in sentence
        tokenize_text = nltk.word_tokenize(sentence)
        #fix formatting issue that might occur.
        if tokenize_text[-1] == '.':
            tokenize_text = tokenize_text[:-1]
        while counter < len(tokenize_text):
            # loops through lexicons titles in lexicon_categories
            for category in lexicon_categories:
                # checks if the first words is in any of the categories
                if tokenize_text[counter] in lexicon_categories[category]:
                    # complete matches is to track all the matches for the following words after first match
                    complete_matches = []
                    # finds maximum match of the keys in the lexicon
                    for word_list in lexicon_categories[category][tokenize_text[counter]]:
                        complete_match = True
                        # use to loop over following words once we find a first match
                        inner_counter = counter + 1
                        # loops through values that are associated with each key in lexicon
                        # if the word exist on its own we consider it as a match
                        if len(word_list) == 0:
                            complete_matches.append(word_list)
                            continue
                        for word in word_list:
                            try:
                                if len(tokenize_text) < inner_counter:
                                    complete_match = False
                                    break
                                # checks to see if all the words in the word_list are matched
                                if tokenize_text[inner_counter] == word:
                                    complete_match = True
                                else:
                                    complete_match = False
                                    break
                            except:
                                logging.exception('could not find a match with word in lexicon, checking next lexicon word')
                                complete_match = False
                                break
                            inner_counter += 1
                        # adds the word_list to complete_matches if all the words match following the first matched
                        # word
                        if complete_match:
                            complete_matches.append(word_list)
                    # check for complete matches and find length of maximum matched key
                    try:
                        # if first word exist on its own and not matched with any following words.
                        if complete_matches == [[]]:
                            max_length = 1
                        else:
                            # finding max length of matched words.
                            max_length = max([len(k) for k in complete_matches])
                    except:
                        max_length = 0
                    # replace the matched with the respective lexicon title
                    if max_length > 0:
                        if complete_matches != [[]]:
                            # perfect match finds the list of words that have the highest matching length
                            perfect_match = [k for k in complete_matches if len(k) == max_length][0]
                            # reaplacing all matched words with the lexicon title.
                            new_tokenize_text = tokenize_text[:counter] + ['<' + category + '>'] + tokenize_text[
                                                                                                   counter + len(
                                                                                                       perfect_match) + 1:]
                        else:
                            # replacing first matched word with lexicon title
                            new_tokenize_text = tokenize_text[:counter] + ['<' + category + '>'] + tokenize_text[
                                                                                                   counter + 1:]
                        tokenize_text = new_tokenize_text.copy()
                    # print(complete_matches, max_length, perfect_match)
            counter += 1
        final_tagged_text = ' '.join(tokenize_text)    
        return final_tagged_text

    def __remove_non_tag_word(self, sentence):
        """
        remove words that are not tags from sentence.
        :param sentence: string representing text that have been transformed by replace_words_with_tags function
        :return: list of lexicon tags
        """
        # remove words that are not tags.
        tags = []
        for i in sentence.split(' '):
            if '<' in i and '>' in i:
                tags.append(i)
        return tags

    def __predict_tags(self, tags, trained_apriori):
        """
        predict sentiment based on tags
        :param tags: list of lexicon titles
        :param trained_apriori: list of dictionaries corresponding to association rules formed during apriori training
        :return: string which is either "Negative","Positive","Neutral"
        """

        for i in trained_apriori:
            # print(i.split(', '))
            if set(tags) == set(i.split(', ')):
                # returns key,value pair with highest value in comparison to other key's value
                result = max(trained_apriori[i].items(), key=operator.itemgetter(1))
                # first element correspond to the key which is 'negative' , 'positive' or 'neutral'
                return result[0]
        return 'neutral'
                    

    def predict_text(self, text):
        """
        predict sentiment given an open text
        :param text: string corresponding to text that want to be analyzed
        :return: string which is either "Negative","Positive","Neutral"
        """
        # predict sentiment of text
        tag = self.__remove_non_tag_word(self.replace_words_with_tags(text, self.TAGGING))
        return self.__predict_tags(tag, self.RULES)

    def predict_text_series(self, texts):
        """
        predict sentiment given a series of open text
        :param texts: series of string corresponding to text that want to be analyzed
        :return: series of string which is either "Negative","Positive","Neutral"
        """
        # predict sentiment of a series of text returning a series
        tagged_text_series = texts.apply(lambda x: self.__remove_non_tag_word(self.replace_words_with_tags(x, self.TAGGING)))
        predict_result = tagged_text_series.apply(lambda x: self.__predict_tags(x, self.RULES))
        return predict_result
