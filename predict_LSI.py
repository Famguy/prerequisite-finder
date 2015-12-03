import glob, os, codecs, string
from gensim import corpora, models, similarities

data_dir = "books/science_"
data = []

tocheckindex = {}

tocheck = glob.glob(data_dir+'9'+'/*.txt')
for tc in tocheck:
	f1 = codecs.open(tc, "r", encoding='utf-8', errors='ignore')
	lines = str(f1.read())
	out = lines.translate(string.maketrans("",""), string.punctuation)
	tocheckindex[len(data)] = tc
	data.append(out.lower())

dictionary = corpora.Dictionary.load('books_LSI.dict')
corpus = corpora.MmCorpus('books_LSI.mm')
lsi = models.LsiModel.load('books_bow.lsi')
tfidf = models.TfidfModel(corpus)

doc = data[4]
vec_bow = dictionary.doc2bow(doc.split())

print tocheckindex[4]

vec_lsi = lsi[vec_bow]
print(lsi.print_topic(max(vec_lsi, key=lambda item: item[1])[0]))

index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
index.save('books_LSI.index')

sims = index[vec_lsi] # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples


