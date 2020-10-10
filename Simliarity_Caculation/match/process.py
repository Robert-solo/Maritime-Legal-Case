# -*- coding: utf-8 -*-
import re
from nltk import stem
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os
import Intervert

class PreProc(object):
    def __init__(self, que_documents, documents):
        self.que_documents = que_documents
        self.documents =  documents
        self.corpus_words = set()
        
        
        
    #增加文档库和查询库
    def AddDocs(self,add_documents):
        self.documents = self.documents + add_documents
    def AddQuerys(self,add_querys):
        self.que_documents = self.que_documents + add_querys
     
    
    #去停用词
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
    
    #普通处理,后添加词语统计
    def Normal(self,):
        que_raw_docs = [self.tokenize(doc) for doc in self.que_documents]
        que_docs = self.Stem_voca(que_raw_docs)
        for qu_doc in que_docs:
            for qu_word in qu_doc:
                self.corpus_words.add(qu_word)
            
        processed_raw_docs = [self.tokenize(doc) for doc in self.documents]
        processed_docs = self.Stem_voca(processed_raw_docs)
        for pro_doc in processed_docs:
            for pro_word in pro_doc:
                self.corpus_words.add(pro_word)
            
        return que_docs, processed_docs
        
    
    
    
    #原始的所有符号:r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    #句子分割      
    def Puncut(self,funP = 1):
        all_doc_slices = []
        pattern = r'\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|\^|&|=|\+|，|。|、|；|‘|’|【|】|·|！|…|（|）' #|\(|\)|,'
        if funP == 1:
            pro_texts = self.documents
        else:
            pro_texts = self.que_documents
            
        for text in pro_texts:
            sentence_list = re.split(pattern,text)
            for idx,sentence in enumerate(sentence_list):
                sentence_list[idx] = sentence.strip()
            raw_sentences = [self.tokenize(doc) for doc in sentence_list]
            sentence_list = self.Stem_voca(raw_sentences)
            sentence_list = list(filter(None, sentence_list))
            all_doc_slices.append(sentence_list)
        return all_doc_slices
    
   
    
    
    #检索倒排索引库
    def FetchDocs(self, c_data, que_docs,rel_docs):
        
        inv_filename = "../support/intervert_" + c_data[:-5] + ".txt"
        if os.path.exists(inv_filename):
            pass
        else:
            IntervertModel = Intervert.Inverted_index()
            IntervertModel.inverted_index(self.documents,inv_filename)
        
        
        #获取倒排索引库
        deal_str_list = []
        inverted_word = {}
        for line in open(inv_filename,'r'): 
            deal_str = line[:-1].split(":")[0]
            deal_str_list = eval(line[:-1].split(":")[1])
            inverted_word[deal_str] = deal_str_list  
        
        docs_have = []  #存储所有查询对应的文档编号集合
        for idx,que_doc in enumerate(que_docs):
            doc_have = []
            for que_word in que_doc:
                if que_word in inverted_word:
                    doc_have.extend(inverted_word[que_word])
                    
            doc_have_list = list(set(doc_have))
            
            if len(doc_have_list) < 30:
                doc_have = doc_have + [i for i in range(30)]
                
            
            doc_have = doc_have + [int(i)-1 for i in rel_docs[idx]]
            doc_have_list = list(set(doc_have) )
            docs_have.append(doc_have_list)
        return docs_have

    
    
    #
    def FetchDocs_for_tr(self, c_data, que_docs,rel_docs):
        
        inv_filename = "../support/tr_inv_" + c_data + ".txt"
        
        #获取倒排索引库
        inverted_word = {}
        for line in open(inv_filename,'r'): 
            deal_str = line[:-1].split(":")
            doc_index_list = eval(deal_str[1])
            for word in deal_str[0].split(" "):
                locations = inverted_word.setdefault(word, [])
                locations.extend(doc_index_list)
        
        docs_have = []  #存储所有查询对应的文档编号集合
        for idx,que_doc in enumerate(que_docs):
            doc_have = []
            for que_word in que_doc:
                if que_word in inverted_word:
                    doc_have.extend(inverted_word[que_word])
                    
            doc_have_list = list(set(doc_have))
            
            if len(doc_have_list) < 40:
                doc_have = doc_have + [i for i in range(40)]
                
            
            #doc_have = doc_have + [int(i)-1 for i in rel_docs[idx]]
            doc_have_list = list(set(doc_have) )
            docs_have.append(doc_have_list)
        return docs_have
    
    
    
    #
    def ex_que(self, all_que_slices):
        
        ex_filename = "../support/exword.txt"
        file  = open(ex_filename,'r')
        extend_dict = {}
        extend_dict = eval(file.read())
        
        deal_que_slices = []
        for que_slice in all_que_slices:
            deal_que_slice = []
            for que_item in que_slice:
                ins_list = []
                for que_word in que_item:
                    if que_word in extend_dict.keys()  :
                        ins_list.extend(extend_dict[que_word])
                        print(extend_dict[que_word])
                que_item.extend(ins_list)
                
                deal_que_slice.append(que_item)
                #deal_que_slice.append(list(set(que_item)))
                
            deal_que_slices.append(deal_que_slice)
            
        #print(extend_dict['parasitic'])
        
        return deal_que_slices
    
    
    
    
 
    