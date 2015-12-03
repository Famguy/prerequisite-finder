import glob
import os
from gensim import corpora, models, similarities
import codecs


data_dir = "books/science_"
dirs = [6,7,8,9]

documents = []
data = []

for d in dirs:
	booklist = glob.glob(data_dir+str(d)+'/*.txt')
	for text in booklist:
		f = codecs.open(text, "r", encoding='utf-8', errors='ignore')
		lines = f.read()
		documents.append(str(lines))

tocheck = glob.glob(data_dir+'10'+'/*.txt')
for tc in tocheck:
	f1 = codecs.open(text, "r", encoding='utf-8', errors='ignore')
	lines = f1.read()
	data.append(str(lines))

# remove common words and tokenize
# stoplist = set('for a of the and to in'.split())

stoplist = []
with open('stopwords.txt', 'rU') as f2:
    for line in f2:
        stoplist.append(line.strip())

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
for text in texts:
	for token in text:
		frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('books.mm', corpus)
mm_corpus = corpora.MmCorpus('books.mm')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(mm_corpus, id2word=dictionary, num_topics=50)
# corpus_lsi = lsi[corpus_tfidf]

# lsi.print_topics(50)

# i = 0
# for topic in lsi.show_topics():
#     print '#' + str(i) + ': ' + topic
#     i += 1

doc = data[0]
vec_bow = dictionary.doc2bow(doc.lower().split())

vec_lda = lda[vec_bow]
print(lda.print_topic(max(vec_lda, key=lambda item: item[1])[0]))

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
vec_lsi = lsi[tfidf[vec_bow]]
print(lsi.print_topic(max(vec_lsi, key=lambda item: abs(item[1]))[0]))


# index = similarities.MatrixSimilarity(lda[corpus])
# index.save('books.index')
# index = similarities.MatrixSimilarity.load('books.index')

# sims = index[vec_lda]
# sims = sorted(enumerate(sims), key=lambda item: -item[1])

# print sims

#for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#	print(doc)

#print(dictionary.token2id)
#print corpus



