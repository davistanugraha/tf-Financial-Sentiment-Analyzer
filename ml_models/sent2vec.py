import spacy
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import string
import re
from ml_models.apriori import Apriori_Model
class Sent2Vec():
    """ Used to create a Sentence2Vector mdoel.
        This implementation is based on Word2Vec, and aggregating the vectors obtained
        per word to represent for sentence

        Public Attributes:
            size (int): Dimensionality of the word vectors
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Use these many worker threads to train the model
            sg (0,1 int): Training algorithm: 1 for skip-gram; otherwise CBOW.
            epochs(int): Number of iterations (epochs) over the corpus.
            stopwords (None, set): NLTK stop words
            stemmer (None, PorterStemmer): NLTK portter Stemmer
            NLP (spacy tokenizer): Tokenizing sentences and words
            TAGGING (dict): contains all lexicons
            w2vmodel (None, word2vec): saved word2vec model
    """
    def __init__(self, size=50, window=5, min_count=5, workers=4, sg = 1, epochs= 50, stopwords = None, stemmer = None):
        # Initializing the model
        self.size = size
        self.window= window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.epochs = epochs
        #loading dictionary for tagging
        self.NLP = spacy.load("en_core_web_sm")
        self.TAGGING = pd.read_pickle("resources/pickle/word_list_dict.pkl")
        #loading word2vec association rules
        self.apriori = Apriori_Model()
        self.w2vmodel = pd.read_pickle("resources/pickle/sent2vec.pkl")

            
    def _remove_symbols(self, text):
        """ Used to remove any symbols from the text
        
        Args:
            text(str): sentence from earnings call or from UI

        returns:
            text(str): sentence with stripped symbols
        """
        text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        return text

    def _replace_numbers(self, text):
        """ Used to replace numbers from the text to tag indicating number
        
        Args:
            text(str): sentence from earnings call or from UI

        returns:
            text(str): sentence with number tags
        """
        return re.sub(r'\b\d+\b', '<number>', text)

    def _convert_lower(self, text):
        """ converts sentence to lowercase
        
        Args:
            text(str): sentence from earnings call or from UI

        returns:
            text(str): converts sentence to lowercase
        """
        return text.lower()

    def _convert_to_tokens(self, text):
        """ Used to convert the text into tokens
        
        Args:
            text(str): sentence from earnings call or from UI

        returns:
            tokens(list): list of words
        """
        tokens = text.split(' ')
        return tokens

    def _remove_stopwords(self, tokens):
        """ Used to remove any stopwords from the text
        
        Args:
            tokens(list): list of words

        returns:
            transformed_tokens(list): list of filtered words 
        """
        if self.stopwords == None:
            return tokens
        transformed_tokens = []
        for word in tokens:
            if word in self.stopwords:
                continue
            else:
                transformed_tokens.append(word)
        return transformed_tokens

    def _stem_words(self, tokens):
        """ Used to stem words
        
        Args:
            tokens(list): list of words

        returns:
            transformed_tokens(list): list of stemmed words
        """
        if self.stemmer == None:
            return tokens
        transformed_tokens = []
        for word in tokens:
            if '<' not in word:
                word = self.stemmer.stem(word)
            transformed_tokens.append(word)
        return transformed_tokens

    def _remove_zero_length_tokens(self, tokens):
        """ Used to remove any tokens with empty strings
        
        Args:
            tokens(list): list of words

        returns:
            filtered_tokens(list): list of filtered words
        """
        filtered_tokens = []
        for token in tokens:
            if len(token.strip()) == 0:
                continue
            else:
                filtered_tokens.append(token.strip())
        return filtered_tokens

    # combining the vectors to represent it for sentences
    def _filter_vocab_sent_aggregate(self, tokens, model= None):
        """ Used to filter the vocabulary to convert the words to vectors,
            Vectors are then aggregated to represent vectors for sentence.
        
        Args:
            tokens(list): list of words

        returns:
            (np.array): list of vector
        """
        filtered_tokens = []
        for word in tokens:
            if word in model.wv.vocab:
                filtered_tokens.append(np.array(model.wv.get_vector(word)))
            else:
                continue
        return np.mean(filtered_tokens, axis=0)
    
        # combining the vectors to represent it for sentences
        
    def _filter_vocab_sent_vec_matrix(self, tokens, model= None):
        """ Used to filter the vocabulary to convert the words to vectors,
            Vectors are then aggregated to represent vectors for sentence.
        
        Args:
            tokens(list): list of words

        returns:
            (np.array): list of vector
        """
        filtered_tokens = []
        for word in tokens:
            if word in model.wv.vocab:
                filtered_tokens.append(model.wv.get_vector(word))
            else:
                continue
        return filtered_tokens
    
    def _preprocess_steps(self, series):
        """ Peforms pre-process operations specific to this model
            Remove symbols
            Replace numbers with number tag
            convert to lowercase
            replace words with tags from lexicon
            convert sentence to tokens
            remove stopwords (if stopwords is loaded when model creation)
            stem words (if stemmer is loaded when model creation)
            remove empty tokens
        
        Args:
            series(Pandas series): sentence from earnings call or from UI

        returns:
            series(Pandas series): series of pre-processed tokens
        """
        # Apply pre-processing to sections_df
        series = series.apply(lambda x: self._convert_lower(self._replace_numbers(self._remove_symbols(x))))
        # Replace with tags
        series = series.apply(lambda x: self._convert_to_tokens(self.apriori.replace_words_with_tags(x, self.TAGGING)))
        # Remove Stop words
        series = series.apply(lambda x: self._remove_stopwords(x))
        # Convert to tokens
        series = series.apply(lambda x: self._stem_words(x))
        # Remove 0 length tokens
        series = series.apply(lambda x: self._remove_zero_length_tokens(x))
        # calculating Sentence vectors
        return series

    def train(self, paragraph_series):
        """ Used to train the model from all the earnings call text 
            and saves trained model to a pickle 
        
        Args:
            paragraph_series(Pandas series): series of text from earnings call by speaker/section
        """
        paragraph_series = paragraph_series.copy()
        preprocessed_series = self._preprocess_steps(paragraph_series)
        self.w2vmodel = Word2Vec(sentences=preprocessed_series, size= self.size, window= self.window, min_count= self.min_count, workers=self.workers, sg = self.sg)
        self.w2vmodel.train(preprocessed_series, total_examples=self.w2vmodel.corpus_count, epochs=self.epochs)
        pickling_on_ = open("resources/pickle/sent2vec.pkl","wb")
        pickle.dump(self.w2vmodel, pickling_on_)
        pickling_on_.close()

    
    def transform_text(self,text):
        """ Used to convert a given text to vectors
        
        Args:
            text(str): text received from UI

        returns:
            sentence_vector(np.array): list of vector
        """
        transformed_text = self._convert_lower(self._replace_numbers(self._remove_symbols(text)))
        tagged_tokens = self._convert_to_tokens(self.apriori.replace_words_with_tags(transformed_text, self.TAGGING))
        # remove stop words
        tagged_tokens = self._remove_stopwords(tagged_tokens)
        tagged_tokens = self._stem_words(tagged_tokens)
        tagged_tokens = self._remove_zero_length_tokens(tagged_tokens)
        sentence_vector = self._filter_vocab_sent_aggregate(tagged_tokens, self.w2vmodel)
        return sentence_vector
    
    def transform_text_to_vec_matrix(self,text):
        """ Used to convert a given text to vectors
        
        Args:
            text(str): text received from UI

        returns:
            sentence_vector(np.array): list of vector
        """
        transformed_text = self._convert_lower(self._replace_numbers(self._remove_symbols(text)))
        tagged_tokens = self._convert_to_tokens(self.apriori.replace_words_with_tags(transformed_text, self.TAGGING))
        # remove stop words
        tagged_tokens = self._remove_stopwords(tagged_tokens)
        tagged_tokens = self._stem_words(tagged_tokens)
        tagged_tokens = self._remove_zero_length_tokens(tagged_tokens)
        sentence_vector = self._filter_vocab_sent_vec_matrix(tagged_tokens, self.w2vmodel)
        return sentence_vector
    
    def transform(self,series):
        """ Used to convert a given text series to vector series
        
        Args:
            series(Pandas series): series of text from earnings call

        returns:
            sentence_vectors(series(np.array)): series of vectors
        """
        series = series.copy()
        preprocessed_series = self._preprocess_steps(series)
        sentence_vectors = preprocessed_series.apply(lambda x: self._filter_vocab_sent_aggregate(x, self.w2vmodel)) 
        return sentence_vectors