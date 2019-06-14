# Financial-Sentiment-Analyzer

Purpose:
This project's purpose is to aid in quantifying financial qualitative data.We are building a sentence classifier that classify sentences of financial context to their respective sentiment (positive,
negative,neutral). Realized that there a lot of publicly available sentiment analyzers, but they are trained on general sentences, and would often misclassify these subject matter sentences.
For example, liability might not be bad in a financial context as they are actually indicators in the financial world. We are building a tool which is able to consider both the semantic and subject matter properties
of a sentence. Hence, "financial tagging" would capture the financial context of the sentence while the "vector embedding" would aid in capturing the "semantic" properties 


2 methods:


  1.Financial Tagging +  Sentence2Vector embedding + classical downstream classifiers
  
  
  2.Financial Tagging +  Word2Vec2vector embedding + convolutional neural network classifier
    
    
# Using this repo:
  There are 3 main parts(each notebook represents one part):
  
    1. Tagging sentences with financial lexicons + Training word2vec and sentence2vec embedding:
          1.0 Parse through real financial earnings call using a pdfparser(built this inhouse, can be found under ml_models,p.s:     yes i know its not an ml_model should have created a file called services but oh well im lazy)
          1.1 break down each call to sentences and replace words that belong to a lexicon to its title(all the lexicons can be found under """resources/tagging_lexicons""", function to do this is embedded in ml_models/apriori.py within the Apriori class) 
          1.2 Feed them to a word2vec model for training(unsupervised),also built a class sent2vec(can be found in ml_models)
          that have preprocessing functions to "clean" sentences, train word2vec models, and form sentence2vectors by aggregating and averaging these word vectors
          
    2. sentence2vec + classical downstream classifiers
        2.1 convert sentences to vectors utilizing the model that we trained in section 1. 
        2.2 feed it to classical ml models
   
    3. word2vec + CNN
        2.1 convert each sentences to a 2d tensor(we fix dimensions of the tensor hence we cut off sentences) utilizing the model andfunctions that we have built and train in sentence2vec class in step 1.
        2.2 construct a 1 layer neural network with 5 different size convolution filters, one drop out, one full layer.
        clear illustraion of netwrok architecture could be found here : https://www.researchgate.net/figure/Illustration-of-our-CNN-model-for-sentiment-analysis-Given-a-sequence-of-d-dimension_fig3_321259272
        
sidenote: a lot of the functions you see in the notebook does not come from open source libraries, you can find all the functions, classes and their explanation wthin ml_models if you wish to do so.              
