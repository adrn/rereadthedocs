import urllib

from bs4 import BeautifulSoup

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


def text_from_url(url):
    url_content = urllib.request.urlopen(url).read()
    text = BeautifulSoup(url_content).get_text()
    return text


moduls = ['wcs', 'vo', 'visualization', 'utils', 'units', 'time', 'table',
          'stats', 'samp', 'nddata', 'modeling', 'io/ascii', 'io/fits',
          'cosmology', 'coordinates', 'convolution', 'constants']

sid  = SentimentIntensityAnalyzer()

for modul in moduls:
    print(modul)
    index_page = text_from_url('http://docs.astropy.org/en/stable/{}/index.html'.format(modul))
    total_score = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    tokens = tokenize.sent_tokenize(index_page)
    for sen in tokens:
        ss = sid.polarity_scores(sen)
        for k in sorted(ss):
            total_score[k] += ss[k]
    print(total_score)
