import urllib
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import matplotlib.pyplot as plt
import numpy as np
'''
Do sentiment analysis on text scraped from a list of websites.

Before running, you must download the nltk resources for vader:
> python
> import nltk
> nltk.download()
'''

def text_from_url(url):
    url_content = urllib.request.urlopen(url).read()                                                                          
    text = BeautifulSoup(url_content, 'html5lib').get_text()
    return text

def measure_sentiment(text):
    '''
    Input: one sentence or small chunk of text
    Result: dict with scores
    '''
    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(text)
    return score

astropydocs = [
               'http://docs.astropy.org/en/stable/constants/index.html',
               'http://docs.astropy.org/en/stable/units/index.html',
               'http://docs.astropy.org/en/stable/nddata/index.html',
               'http://docs.astropy.org/en/stable/table/index.html',
               'http://docs.astropy.org/en/stable/time/index.html',
               'http://docs.astropy.org/en/stable/coordinates/index.html',
               'http://docs.astropy.org/en/stable/wcs/index.html',
               'http://docs.astropy.org/en/stable/modeling/index.html',
               #'http://docs.astropy.org/en/stable/analytic_functions/index.html'
               'http://docs.astropy.org/en/stable/io/unified.html',
               'http://docs.astropy.org/en/stable/io/fits/index.html',
               'http://docs.astropy.org/en/stable/io/ascii/index.html',
               'http://docs.astropy.org/en/stable/io/votable/index.html',
               'http://docs.astropy.org/en/stable/io/misc.html',
               'http://docs.astropy.org/en/stable/samp/index.html',
               'http://docs.astropy.org/en/stable/vo/index.html',
               'http://docs.astropy.org/en/stable/cosmology/index.html',
               'http://docs.astropy.org/en/stable/convolution/index.html',
               'http://docs.astropy.org/en/stable/visualization/index.html',
               'http://docs.astropy.org/en/stable/stats/index.html',
               'http://docs.astropy.org/en/stable/config/index.html',
               'http://docs.astropy.org/en/stable/io/registry.html',
               'http://docs.astropy.org/en/stable/logging.html',
               'http://docs.astropy.org/en/stable/warnings.html',
               'http://docs.astropy.org/en/stable/utils/index.html',
               'http://docs.astropy.org/en/stable/testhelpers.html',
               #'http://docs.astropy.org/en/stable/development/workflow/get_devel_version.html']
               ]

allnames = []
allresults = []
for doc in astropydocs:
    if 'index' in doc:
        name = doc[34:-11]
    else:
        name = doc[34:-5]
    result = 0
    webtext = text_from_url(doc)
    sentences = tokenize.sent_tokenize(webtext)
    for sentence in sentences:
        score = measure_sentiment(sentence)
        result += score['compound']
    result = result/len(sentences)
    print(name, result)
    allnames.append(name)
    allresults.append(result)

fig, ax = plt.subplots()
y_pos = np.arange(len(allnames))
ax.barh(y_pos, allresults, align='center', color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(allnames, size='small')
ax.set_xlabel('Compound score')
plt.show()