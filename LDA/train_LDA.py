import glob, os, codecs, string
from gensim import corpora, models, similarities

data_dir = "books/science_"
dirs = [6,7,8,9]

documents = []
data = []
bookindex = {}

# remove punctuation and add to document directory

for d in dirs:
	booklist = glob.glob(data_dir+str(d)+'/*.txt')
	for ind, text in enumerate(booklist):
		f = codecs.open(text, "r", encoding='utf-8', errors='ignore')
		lines = str(f.read())
		out = lines.translate(string.maketrans("",""), string.punctuation)
		bookindex[len(documents)] = text
		documents.append(out.lower())
		

# remove common words and tokenize

stoplist = []
with open('stopwords.txt', 'rU') as f2:
    for line in f2:
        stoplist.append(line.strip())

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)

texts = [[word for word in document.split() if word not in stoplist] for document in documents]
for text in texts:
	for token in text:
		frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# store to disk, for later use
corpora.MmCorpus.serialize('books_LDA.mm', corpus) 
dictionary.save('books_LDA.dict')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=20) # initialize an LDA transformation

# print topics
i = 0
for topic in lda.show_topics():
    print '#' + str(i) + ': ' + topic
    i += 1

lda.save('books.lda')

# select top 20 words for each of the 20 LDA topics
top_words = [[word for _, word in lda.show_topic(topicno, topn=20)] for topicno in range(lda.num_topics)]
print(top_words)



import json
with open('dataLDA.json', 'w') as fp:
    json.dump(bookindex, fp)