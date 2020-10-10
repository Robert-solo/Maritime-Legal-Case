#!/usr/bin/env python
# -*- coding: utf-8 -*-


#常用短语提取
from nltk import stem
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


class Inverted_index(object):
    def __init__(self):
        
        self.inverted = {}
        self.processed_raw_docs = []
        self.processed_docs = []
        self.initialize()
    def initialize(self):
        pass
    #单词预处理功能
    #过滤停放用词
    def tokenize(self,text):
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]
    
    #提取词根
    def Stem_voca(self,processed_docs):
        WordNetStem = stem.WordNetLemmatizer()
        for idx,words in enumerate(processed_docs):
            for i,word in enumerate(words):
                words[i] = WordNetStem.lemmatize(word,"n")
            for i,word in enumerate(words):
                words[i] = WordNetStem.lemmatize(word,"v")
                #processed_docs[idx] = words
            processed_docs[idx] = [token for token in words if token not in STOPWORDS]
        return processed_docs
    
    def word_split(self,documents):
        self.processed_raw_docs = [self.tokenize(doc) for doc in documents]
        self.processed_docs = self.Stem_voca(self.processed_raw_docs)
        return self.processed_docs
        
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
    
    def inverted_index_add(self, doc_id, doc): 
        for word in self.tokenize(doc): 
            locations = self.inverted.setdefault(word, [])
            locations.append(doc_id)
        return self.inverted





    