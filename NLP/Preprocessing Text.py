"The Goal of this project is to pretty much automate the tedious steps of cleaning text data"
#Start Date: 13th August 2019
#Source 1: https://stackoverflow.com/questions/30315035/strip-numbers-from-string-in-python
#Source 2: https://medium.com/@pemagrg/pre-processing-text-in-python-ad13ea544dae
#Source 3: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
#Source 4: https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
#Source 5: https://stackoverflow.com/questions/12628958/remove-small-words-using-python
#Source 6: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
#Source 7: https://www.programcreek.com/python/example/107282/nltk.stem.WordNetLemmatizer
#Source 8: https://www.w3schools.com/python/python_howto_remove_duplicates.asp
#Source 9: https://stackoverflow.com/questions/1546226/simple-way-to-remove-multiple-spaces-in-a-string
#Source 10: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
#Source 11: https://stackoverflow.com/questions/9233027/unicodedecodeerror-charmap-codec-cant-decode-byte-x-in-position-y-character
#Source 12: https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text%20Normalization%20Demo.ipynb
#Source 13: https://python.gotrained.com/frequency-distribution-in-nltk/
text = """   Harry Potter" is the most miserable, lonely boy you can imagine. He's shunned by his relatives,
          the Dursley's, that have raised him since he was an infant. He's forced to live in the cupboard
          under the stairs, forced to wear his cousin Dudley's hand-me-down clothes, and forced to go to his
          neighbour's house when the rest of the family is doing something fun. Yes, he's just about as miserable
          as you can get. 1234123 !@#$%^&*()____.</body></head>"""



#To read in a text file as a single string for cleaning later on:
#The following code is called a context manager, which saves us processing time.
with open("C:/Users/miqui/OneDrive/ML-DL-Datasets/TheBeast by Lovecraft.txt", 'r') as file:
    beast = file.read().replace('\n', '')
len(beast)


########################Need to figure out how to parse this text file!!!!!!!!!!
*All you have to do was specify the encoding at "utf-8" \

with open("C:/Users/miqui/OneDrive/ML-DL-Datasets/Frankenstein.txt", "r", encoding = "utf8") as file:
    Frankenstein = file.read().replace("\t", '')
len(Frankenstein) #420,020 characters


#Notes:
***"I think you have to remove the HTML before the punctuation, otherwise" \
   "body and head will still be in the cleaned dataset"

***I'm wondering if I should stem the words, but that may lose me some contextual information"

***I think I want to remove the stop words first and then lemmatize the text

***I did it!
***I suppose it would be wise to remove the duplicate words from the list of words in the output
***But make it a user input because what if i want to count the word frequency
***I had to update the stopwords list to remove "-PRON-" for some reason
***I spent a good couple of hours trying to find an easier way to lemmatize the words, or at least make it look cleaner

***I need to figure out a way to read in the Frankenstein file, but it says that I have a UnicodeDecodeError
    ***Need to use the "utf-8" encoding scheme

#Actual cleaning function:
def Clean_Text(text, language, MinWordLength=2, lemma=False, keepShort=True, keepDuplicates=True):
    """
    This function was compiled to quickly clean up messy text data
    It's only input can be a string, so you must first convert the txt document to a string
    :param text: The input text that you would like to clean and tidy up
    :param language: Required; The language you want your stopwords to take
    :param MinWordLength: If keepShort = True, then this is the minimum length of a word that you'll get in your words list
    :param lemma: If lemma = True, then this will lemmatize the words
    :param keepShort: Boolean to decide if you want to keep words of a certain length or less
    :param keepDuplicates: Boolean to decide if you want to keep duplicate words in your word list
    :return:
    """
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize.regexp import regexp_tokenize
    import spacy
    import unicodedata
    text = re.sub('<[^<]+?>', '', text) #Remove HTML
    text = re.sub("[^a-zA-Z]", " ", text) #Remove punctuation
    text = re.sub('[0-9]+', " ", text) #Remove numbers
    text = text.strip()     #Remove whitespaces
    text = re.sub(" +", " ", text) #Remove extra spaces
    text = text.lower()     #Lowercase text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore") #Remove accents
    stopword = stopwords.words(language) # Create List of noisy/not helpful words
    stopword.append("-PRON-")
    if lemma == True: #Lemmatize words
        nlp = spacy.load("en_core_web_sm") #Load the main SpaCy model
        doc = nlp(text)
        docs = " ".join([token.lemma_ for token in doc]) #Lemmatize and Tokenize
        words = regexp_tokenize(docs, pattern='\s+', gaps=True)
    words = [word for word in words if word not in stopword] #Removing stopwords
    if keepShort == False:
       words = [word for word in words if len(word) >= MinWordLength] #Remove words of variable length
    if keepDuplicates == False:
       words = list(dict.fromkeys(words)) #Remove duplicate words
    return text, words


#Okay I'm officially stuck.
#Some NLP requires tokenization while others require whole, clean sentences
#I wonder if there's a way to have it so you can tell the program which output
#you would like to return, the clean sentences or the clean words

clean = Clean_Text(text = Frankenstein,
                   language = "English",
                   lemma = True,
                   keepShort = False,
                   keepDuplicates = True,
                   MinWordLength = 4)

clean_text = clean[0]
clean_words = clean[1]
#It looks like you get something called a tuple as a result of running the function.
#Now I'm wondering how you access just the words or just the sentences from the tuple
    #Okay, it looks like you can access just the words or just the sentences by using an index
    #In Python, indices start at 0, so to get the sentences, just use the 0 index
    #For the cleaned words, just use the 1 index

#Frequency of words, assuming keepDuplicates = True
import collections
counter = collections.Counter(clean_words)
print(counter)

#To plot the frequency of the most common words
freqDist = FreqDist(clean_words)
freqDist.plot(12)



#WordCloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 900, height = 900,
                      min_font_size = 10,
                      max_words = 50, stopwords = stopwords,
                      background_color = "black",
                      relative_scaling = 1).generate(clean[0])
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


#Important word frequencies
import collections
freqs = collections.Counter(clean[1])
freqs.most_common(n = 30)



#Sentiment Analysis of an H.P. Lovecraft story:
beast_clean = clean[0]
def sentiment_analyzer_scores(sentence):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(sentence)
    print(score)
sentiment_analyzer_scores(beast_clean)
#Very low sentiment, quite a dark story!


"To clean multiple texts in a dataframe quickly"
def Clean_DF_Text(text):
    import re
    from nltk.corpus import stopwords
    import unicodedata
    text = re.sub('<[^<]+?>', '', text) #Remove HTML
    text = re.sub("[^a-zA-Z]", " ", text) #Remove punctuation
    text = re.sub('[0-9]+', " ", text) #Remove numbers
    text = text.strip()     #Remove whitespaces
    text = re.sub(" +", " ", text) #Remove extra spaces
    text = text.lower()     #Lowercase text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore") #Remove accents
    stopwords = stopwords.words('english')
    words = text.split()
    clean_words = [word for word in words if word not in stopwords]
    text = ' '.join(clean_words)
    return text


#To extract just the numbers from a string
def only_numbers(text):
    import re
    text = re.sub("[^0-9]", "", text)
    return text
df["Numeric"] = df["Text"].apply(only_numbers)

wines = pd.read_csv("wine.csv")
wines = wines[["points","description", "title"]]
wines.shape
wines["clean"] = wines["description"].apply(Clean_DF_Text)
wines["year"] = wines["title"].apply(only_numbers)