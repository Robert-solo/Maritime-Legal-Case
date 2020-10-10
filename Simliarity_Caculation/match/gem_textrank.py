#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 02:48:38 2019

@author: leyv
"""

import gensim
from six import iteritems
from gensim.summarization import summarize
from gensim.summarization import keywords
import fileDeal as fd
import process as pp


from nltk import stem
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

#提取词根
def Stem_voca(processed_docs):
    WordNetStem = stem.WordNetLemmatizer()
    for idx,words in enumerate(processed_docs):
        for i,word in enumerate(words):
            words[i] = WordNetStem.lemmatize(word,"n")
        for i,word in enumerate(words):
            words[i] = WordNetStem.lemmatize(word,"v")
            #processed_docs[idx] = words
        processed_docs[idx] = [token for token in words if token not in STOPWORDS]
    return processed_docs






#c_data = "npl_data"
c_data = "med_data"
#c_data = "LISA_data"
tdata = fd.FileDeal(c_data)



#记录候选文档和文档编号为
documents = tdata.documents
file_index = tdata.file_index  
#print(file_index[3809])
#记录查询文本和查询编号为
que_documents = tdata.que_documents
que_index = tdata.que_index
#结果集对应查询和答案文本编号为
rel_index = tdata.rel_index
rel_docs = tdata.rel_docs
perwords = sum([len(doc) for doc in documents])/len(documents)


processed_raw_docs = [tokenize(doc) for doc in documents]
processed_docs = Stem_voca(processed_raw_docs)



print(processed_docs[0])
#print(len(documents))
#test_str = 'PERSONAL VIEW OF THE SERVICE, BASED ON OBSERVATIONS MADE DURING WORK UNDER THE VOLUNTARY SERVICES OVERSEAS SCHEME. IN KASAMA, THE PROVINCIAL CENTRE OF THE NORTHERN PROVINCE, 3 MAJOR PROBLEMS FACED LIBRARY SERVICES-POOR TRANSPORT; THE LEGACY LEFT BY AN ACTING LIBRARIAN WHO HAD OPERATED HIS OWN KIND OF LIBRARY SERVICE WHICH WAS EXTREMELY CONFUSED ; AND THE LACK OF PROVISION FOR THE PRACTICAL TRAINING OF NEWLY QUALIFIED STAFF. IMPROVEMENTS NEED TO BE MADE TO THE SYSTEM IN TERMS OF AUTHORITY, DEVELOPMENT, TRANSPORT, TRAINING, PROFESSIONAL STATUS AND PROMOTIONAL ACTIVITIES. '
#print(test_str.lower())
#test_str = test_str.lower()
#summary_result = summarize(test_str,word_count= 100,ratio=0.5 ,split= True)
#print(summary_result)

#wordlist = keywords(test_str,ratio=0.8,pos_filter = ['NN', 'JJ' , 'VB'], scores=True, split=True, lemmatize= True)

num2dict = set()
num3dict = set()
num4dict = set()
inverted = {}
for index,test_str in enumerate(documents):
    test_str = test_str.lower()
    wordlist = keywords(test_str,ratio=0.8, pos_filter = ['NN', 'JJ' , 'VB'], scores=True, split=True, lemmatize= True)
    #print(wordlist)
    for word_item in wordlist:
        word = word_item[0]
        
        '''
        if len(word.split(' ')) == 2:
            num2dict.add(word)
        if len(word.split(' ')) == 3:
            num3dict.add(word)
        if len(word.split(' ')) == 4:
            num4dict.add(word)
        '''
        #locations = inverted.setdefault(word, []) 
        #locations.append((index,word_item[1]))
        
        if word not in STOPWORDS:
            locations = inverted.setdefault(word, []) 
            locations.append(index) 
       
        
    print(index,':',len(wordlist)) # 保留80%试试

print("total words",len(inverted))

print("two words",':',len(num2dict)) 
print("three words",':',len(num3dict))
print("four words",':',len(num4dict))


file = open("../support/tr_inv_" + c_data + ".txt",'w')
for word,list_item in inverted.items():
    file.write(str(word)+':'+str(list_item)+'\n')
file.close()






'''
 #生成文档分布
    def save_in(self,inv_filename):
        file = open(inv_filename,'w')
        for word,list_item in self.inverted.items():
            file.write(str(word)+':'+str(list_item)+'\n')
        file.close()
        print("保存文件成功")
        return
        
    
    def inverted_index(self,documents,inv_filename): 
        processed_docs = self.word_split(documents)
        for index,processed_doc in enumerate(processed_docs):
            for word in list(set(processed_doc)): 
                locations = self.inverted.setdefault(word, []) 
                locations.append(index) 
        self.save_in(inv_filename)
        return self.inverted 
'''

#text, ratio=0.2, words=None, split=False, scores=False, pos_filter=('NN', 'JJ'),lemmatize=False, deacc=True

'''
import gensim.summarization.keywords as gsk
gsk.get_graph()
#key_g = gensim.summarization.keywords.keyword(test_str)

'''

'''
tokens = gensim.summarization.textcleaner.clean_text_by_word(test_str)

for word, unit in iteritems(tokens):
    print(word," 的词性:", unit)
'''
#print(keywords._clean_text_by_word(test_str) )



