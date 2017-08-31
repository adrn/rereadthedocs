import urllib
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(text)
    return score

astropydocs = ['http://docs.astropy.org/en/stable/constants/index.html',
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
               'http://docs.astropy.org/en/stable/development/workflow/get_devel_version.html']

for doc in astropydocs:
    #print(doc)
    result = measure_sentiment(text_from_url(doc))
    print(doc[34:-11], result['compound'])

