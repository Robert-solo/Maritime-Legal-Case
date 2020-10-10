# -*- coding: utf-8 -*-
from nltk.corpus import wordnet as wn

from nltk import stem

word = "apple"
word = "surgery"
word = "surgical"
word = "palliation"
word = "palliate"
word = "technology"
word = "technique"

WordNetStem = stem.WordNetLemmatizer()
print(WordNetStem.lemmatize(word,'n'))



qe_word_list = []

ss = wn.synsets(word)
print(ss)
print(ss[0].lemma_names())

for synset in ss:
    qe_word_list.extend(synset.lemma_names())
    
result = list(set(qe_word_list))
print(result)




# -*- coding: utf-8 -*-
import fileDeal as fd
import process as pp

'''
0.重要参数
'''
#使用何种数据
#c_data = "npl_data"
c_data = "med_data"
#c_data = "LISA_data"

#使用何种方法
#Match_fun = "wordSim1"
#Match_fun = "wordSim2"
Match_fun = "bm25"
#Match_fun = "wmd"
#Match_fun = "lmj"


w2v_kind = 1
word2vec = []
if Match_fun == "wmd" or Match_fun == "wordSim1" or Match_fun == "wordSim2":
    import gensim
    if w2v_kind == 1:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove-wiki-gigaword-100/glove-wiki-gigaword-100.txt',binary=False)
    elif w2v_kind == 2:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove.6B/glove.6B.100d.txt',binary=False)
    elif w2v_kind == 3:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/word2vec-google-news-300/GoogleNews-vectors-negative300.bin',binary=True)


'''
1.数据提取
'''
tdata = fd.FileDeal(c_data)


#记录候选文档和文档编号为
documents = tdata.documents
file_index = tdata.file_index  
#记录查询文本和查询编号为
que_documents = tdata.que_documents
que_index = tdata.que_index
#结果集对应查询和答案文本编号为
rel_index = tdata.rel_index
rel_docs = tdata.rel_docs

print("In the doc-text dataset there are", len(documents), "textual documents")
print("In the query-text dataset there are ",len(que_index),"query documents")
print("In the rlv-ass dataset there are ",len(rel_docs),"answers")


'''
print(file_index[:3],len(file_index))
print(documents[:3])
print(que_index[:3],len(que_index))
print(que_documents[:3])

print(rel_index,len(rel_index))
print(rel_docs)
'''

'''
2.数据预处理
'''
#all_que_slices  all_doc_slices 为按句子切分的查询候选片段   que_docs processed_docs 为一般化处理的查询候选文档
preprocess = pp.PreProc(que_documents,documents)
all_que_slices = preprocess.Puncut(0)
all_doc_slices = preprocess.Puncut(1)
que_docs,processed_docs = preprocess.Normal()


#print(all_doc_slices[:3])
print(all_que_slices[:3])

#all_que_slices = all_que_slices[:3]
#que_docs = que_docs[:3]

docs_have = preprocess.FetchDocs(c_data, que_docs)

































