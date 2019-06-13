# Financial-Sentiment-Analyzer

Purpose:
This project's purpose is to aid in quantifying financial qualitative data.We are building a sentence classifier that are of financial context to their respective sentiment (positive,
negative,neutral). We realize that there a lot of publicly available sentiment analyzers, but they are trained on general sentences, and would often misclassify these subject matter sentences.
For example, liability might not be bad in a financial context as they are actually indicators in the financial world. We are building a tool which is able to consider both the semantic and subject matter properties
of a sentence. Hence we believe that the "financial tagging" would capture the financial context of the sentence while the embedding would aid in capturing the "semantic" properties 

2 methods:

  1.Financial Tagging +  Sentence2Vector embedding + classical downstream classifiers
  2.Financial Tagging +  Word2Vec2vector embedding + convolutional neural network classifier
    
