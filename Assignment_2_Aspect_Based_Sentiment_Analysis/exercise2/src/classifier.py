import numpy as np
import pandas as pd
import spacy
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class Classifier:
    """The Classifier"""


    #############################################
    def __init__(self):
    	self.spacy_parser = None
    	self.tokenizer = None
    	self.label_encoder_sentiment = None
    	self.sentiment_model = None
    	self.vocab_size = None
    	self.num_sentiments = 3
    	self.num_aspect_categories = 12


    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        col_names = ['Polarity','Aspect_Category','Specific_Target_Aspect_Term','Character_Offset','Sentence']
        train_df = pd.read_csv(trainfile,sep='\t',names=col_names)
        

        self.spacy_parser = spacy.load('en')
        self.vocab_size = 8000
        self.num_aspect_categories = 12 # There are 12 Aspect Categories
        self.num_sentiments = 3 # Positive, Negative and Neutral

        #Drop the character Offset Dataframe
        train_df = train_df.drop(columns=['Character_Offset'])
        
        #Extract Sentiment for Aspect
        Specific_Target_Sentiment_Term = []

        for review in self.spacy_parser.pipe(train_df['Sentence']):
            if review.is_parsed:
                Specific_Target_Sentiment_Term.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                Specific_Target_Sentiment_Term.append('')  

        train_df['Specific_Target_Sentiment_Term'] = Specific_Target_Sentiment_Term

        #Building Bag of Words Representation
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(train_df.Sentence)

        #Building the sentiment analysis model
        self.sentiment_model = Sequential()
        self.sentiment_model.add(Dense(512, input_shape=(self.vocab_size,), activation='relu'))
        self.sentiment_model.add(Dense(self.num_sentiments, activation='softmax'))
        self.sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tokenized_sentiment = pd.DataFrame(self.tokenizer.texts_to_matrix(train_df.Specific_Target_Sentiment_Term))

        # Encoding the Sentiment or Polarity Labels
        self.label_encoder_sentiment = LabelEncoder()
        sentiment_integer_category = self.label_encoder_sentiment.fit_transform(train_df.Polarity)
        sentiment_dummy_category = to_categorical(sentiment_integer_category)

        # Training the Sentiment Model
        history = self.sentiment_model.fit(tokenized_sentiment, sentiment_dummy_category, epochs=5, verbose=1,validation_split=0.3)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('Training_Phase_Accuracy_Curve.png')

        return self

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        col_names = ['Polarity','Aspect_Category','Specific_Target_Aspect_Term','Character_Offset','Sentence']
        dev_df = pd.read_csv(datafile,sep='\t',names=col_names)
        dev_aspect_category = list(dev_df.Aspect_Category)

        #Drop the character Offset Dataframe
        dev_df = dev_df.drop(columns=['Character_Offset'])
                             
        # Sentiment preprocessing
        Dev_Specific_Target_Sentiment_Term = []

        for review in self.spacy_parser.pipe(dev_df['Sentence']):
            if review.is_parsed:
               Dev_Specific_Target_Sentiment_Term.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
               Dev_Specific_Target_Sentiment_Term.append('') 
            
        Dev_Specific_Target_Sentiment_Term = pd.DataFrame(self.tokenizer.texts_to_matrix(Dev_Specific_Target_Sentiment_Term))

        # Models output
        Dev_predict_sentiment = self.label_encoder_sentiment.inverse_transform(self.sentiment_model.predict_classes(Dev_Specific_Target_Sentiment_Term))

        #for i in range(len(dev_aspect_category)):
        	#print("Sentence " + str(i+1) + " is expressing a  " + Dev_predict_sentiment[i] + " opinion about " + dev_df.Aspect_Category[i])

        #Predicted Polarity or Sentiment
        slabels = list(Dev_predict_sentiment)

        return slabels