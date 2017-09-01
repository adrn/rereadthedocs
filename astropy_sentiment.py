import glob
import urllib

from bs4 import BeautifulSoup

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

html_docs_location = '/Users/bsipocz/munka/devel/astropy/docs/_build/html/'


def text_from_url(url):
    url_content = urllib.request.urlopen(url).read()
    text = BeautifulSoup(url_content).get_text()
    return text


def text_from_file(filename):
    file_content = open(filename).read()
    text = BeautifulSoup(file_content).get_text()
    return text


moduls = ['wcs', 'vo', 'visualization', 'utils', 'units', 'time', 'table',
          'stats', 'samp', 'nddata', 'modeling', 'io/ascii', 'io/fits',
          'cosmology', 'coordinates', 'convolution', 'constants']

sid  = SentimentIntensityAnalyzer()

for modul in moduls:
    #page = text_from_url('http://docs.astropy.org/en/stable/{}/index.html'.format(modul))
    total_score = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    docs_pages = (glob.glob('{}/{}/*.html'.format(html_docs_location, modul)) +
                  glob.glob('{}/api/astropy.{}.*html'.format(html_docs_location, modul)))

    for docs_page in docs_pages:
        page = text_from_file(docs_page)
        tokens = tokenize.sent_tokenize(page)
        summa = 0
        score = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        for sen in tokens:
            ss = sid.polarity_scores(sen)
            for k in sorted(ss):
                score[k] += ss[k]
            summa += 1
        if summa > 0:
            total_score = {i: total_score[i]+score[i]/summa for i in total_score}
    if len(docs_pages) > 0:
        total_score = {i: total_score[i]/len(docs_pages) for i in total_score}
    print(modul, total_score)
