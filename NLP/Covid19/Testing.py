import re
from nltk.corpus import stopwords

text = "https://t.co/prmHgbkOXb"
text2 = "Hello https://www.google.com There"
text3 = "hello #corona #coronavirus"
re.sub(r'http\S+', " ", text2)  # Remove urls

re.findall(r"#(\w+)", text3)  # Find hashtags











