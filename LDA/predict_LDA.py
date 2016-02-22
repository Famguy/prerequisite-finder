import glob, os, codecs, string
from gensim import corpora, models, similarities

data_dir = "books/science_"
data = []

tocheck = glob.glob(data_dir+'10'+'/*.txt')
for tc in tocheck:
	f1 = codecs.open(tc, "r", encoding='utf-8', errors='ignore')
	lines = str(f1.read())
	out = lines.translate(string.maketrans("",""), string.punctuation)
	data.append(out.lower())

dictionary = corpora.Dictionary.load('books_LDA.dict')
corpus = corpora.MmCorpus('books_LDA.mm')
lda = models.LdaModel.load('books.lda')
tfidf = models.TfidfModel(corpus)

doc = data[0]
vec_bow = dictionary.doc2bow(doc.split())

vec_lda = lda[tfidf[vec_bow]]
print(lda.print_topic(max(vec_lda, key=lambda item: item[1])[0]))

index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LDA space and index it
index.save('books_LDA.index')

sims = index[vec_lda] # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples



