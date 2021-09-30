import pandas as pd
import re
import unicodedata
import nltk
from wordcloud import WordCloud, STOPWORDS

data = pd.read_excel('data.xlsx')

additional_stopwords = []

def basic_clean(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + additional_stopwords
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


data_base = pd.read_csv('base_ngram.csv')
data_base = pd.DataFrame(data_base)
data_base['word'] = data_base['word'].astype(str).str.strip('''(',)' ''')

for (columnName, columnData) in data.iteritems():
    columnData = columnData.dropna()
    words = basic_clean(''.join(str(columnData.tolist())))
    results = pd.Series(nltk.ngrams(words, 1)).value_counts()
    results = pd.DataFrame(results)
    results.reset_index(inplace=True)
    results['word'] = results['index'].astype(str).str.strip('''(',)' ''')
    results = results.iloc[:, 1:3]
    data_base = pd.merge(data_base, results, how='left', on='word', copy=False)

data_base.columns = ['word', '1900', '1901', '1902', '1903', '1904', '1905', '1906', '1907', '1908', '1909', '1910',
                     '1911', '1912', '1913', '1914', '1915', '1916', '1917', '1918', '1919', '1920', '1921', '1922',
                     '1923', '1924', '1925', '1926', '1927', '1928', '1929', '1930','1931', '1932', '1933', '1934',
                     '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', 
                     '2013', '2014', '2015', '2016', '2017', '2018', ]

data_base = data_base.fillna(0)
data_base.to_csv('input_data_final.csv', index=False)
