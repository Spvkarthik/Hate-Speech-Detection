# -*-#import the libraries
import pandas as pd
# load the dataset
df=pd.read_csv(r"C:\Users\S.Nagesaararao\Desktop\karthik BE\twitter.csv")
df['label']=df['class'].map({0:'hate speech',
                            1:'offensive language',
                            2:'neither hate nor offensive '})
df.isnull().sum() # dats doesn't have any null values
dataset=df[['tweet','label']]

#import the data cleaning algorithms
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import html
stop_words=set(stopwords.words('english'))
wordnet=WordNetLemmatizer()
ps=PorterStemmer()
sb=SnowballStemmer('english')
#user define function to clean the tweet data
def data_cleaning(text):
    text=re.sub(r'@\w+', ' ', text)
    text= re.sub(r'\bRT\b', ' ', text)
    text=html.unescape(text)
    text= re.sub(r'http[s]?://\S+', '', text)
    text=re.sub('[^a-zA-Z]', ' ', text)
    text=text.lower()
    text=text.split()
    text=[wordnet.lemmatize(word) for word in text if not word in stop_words]
    text=' '.join(text)
    return text

dataset['tweet']=dataset['tweet'].apply(data_cleaning)
#till now data cleaning is done
#modelling the dataset
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000, max_df=0.95, min_df=5)
x=cv.fit_transform(dataset['tweet']).toarray()
y=dataset['label'].values
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
models={
        'DecisionTreeClassifier':DecisionTreeClassifier(),
        'LogisticRegression':LogisticRegression(),
        'KNeighborsClassifier':KNeighborsClassifier(),
        'Bernoulli Naive Bayes':BernoulliNB(),
        'Gaussian Naive Bayes':GaussianNB(),
        'Multinomial Naive Bayes':MultinomialNB()
        }
results=[]
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    cm=confusion_matrix(y_test, y_pred)
    ac=accuracy_score(y_test, y_pred)
    bias=model.score(x_train,y_train)
    variance=model.score(x_test, y_test)
    results.append(
        {
            'Name of the model':name,
            'confusion matrix':cm,
            'accuracy(%)':ac*100,
            'bias(%)':bias*100,
            'variance(%)':variance*100
            })
df1=pd.DataFrame(results)
df1.to_csv("Model accuracy results.csv",index=False)


''' from  the above data set either Logistic regression or decision tree classifier
so for this let us consider Logistic regression. Decision tree classifier is present because
of its high interapretability.
Logistic Regression:

Pros: High accuracy, low variance, good for binary classification problems like hate speech detection.
Cons: May not capture complex patterns as well as some other models.
Use Case: Suitable if you need a balance between performance and simplicity, 
and if you can afford to trade off a bit of interpretability for higher accuracy.

Decision Tree Classifier:

Pros: Highly interpretable, can capture complex patterns.
Cons: High bias, which can be problematic for generalization.
Use Case: Good if interpretability is crucial and you need to understand the decision process for each classification. 
However, you might need to prune the tree to avoid overfitting.'''

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred1=dt.predict(x_test)

sample='i will kill you'
sample=data_cleaning(sample)
x=cv.transform([sample]).toarray()
dt.predict(x)