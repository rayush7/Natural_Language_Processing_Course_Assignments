Natural Language Processing Course - Assignment 2 - Aspect Based Sentiment Analysis

1) Authors : Benoit Laures (Email : benoit.laures@student.ecp.fr)
          Ayush K. Rai (Email : ayush.rai2512@student-cs.fr)
          Paul Asquin (Email : paul.asquin@student.ecp.fr)

2) Description of the Final Pipeline

   Download the 100 dimensional pretrained Glove Embeddings :  http://nlp.stanford.edu/data/glove.6B.zip

   In order to attack the problem of Aspect Based Sentiment Analysis, we use the following strategy

   a) We use the Parser from the Spacy Library to extract Adjective, Verb, Noun, Pronoun  and interjection from the sentences, which we call sentiment specific words. The intuition behind this is that these figures of speech capture information about the corresponding Aspect category.
   b) The next step in our pipeline involves transformation of the extracted sentiment specific words into the feature vector by using 100 dimensional pretrained Glove Embeddings. For this we create a non trainable embedding layer using Keras library.
   c) Finally we applied multinomial logistic regression machine learning model for classification. This was a difficult choice for us and we performed alot of experiments. But highly complex models like lstm based models, conv1D based model, fully connected model etc did not outperform the multinomial logistic regression model. Therefore following the Occam's Razor principle, we choose the Multinomail Logistic Regression model.


3) Accuracy on the Dev Set : 0.8138

Additional Note

Other Models we tried which didn't perform well

1) Using Spacy Parser to extract the adjectives and verbs followed by feature extraction using Bag of Words Model and finally applying fully connected neural network for classification. We achieved the accuracy of around 0.773 using this model.

2) Another approach we tried is to extract sentence level features using pretrained Glove Embeddings and then apply multiple machine/deep learning models like Logistic Regression, Conv1D, Random Forest foraspect based sentiment analysis. However we achieved a highest accuracy of 0.78 by using these models.