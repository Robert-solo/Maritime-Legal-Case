# -*- coding: utf-8 -*-

import re


class FileDeal(object):
    def __init__(self, filename):

        self.documents = []
        self.file_index = [] 
        self.que_documents = []
        self.que_index = []
        self.rel_index = []
        self.rel_docs = []

        
        self.steps_query = 0
        
        #判断使用的数据集
        if filename == "npl_data":
            self.fetch_npl_data()
        elif filename == "med_data":
            self.fetch_med_data()
        elif filename == "LISA_data":
            self.fetch_LISA_data()
        
            
    
    #nlp数据集
    def fetch_npl_data(self):
        
        #抽取文档
        filename = '../dataset/raw/doc-text'
        file = open(filename)
        contents = file.read()
        self.documents = re.split('/\n',contents[2:-1])
        for idx,doc_item in enumerate(self.documents):
            index_doc = doc_item.split('\n',1)
            self.file_index.append(index_doc[0])
            self.documents[idx] = index_doc[1].replace('\n',' ').strip()
        #print("In the doc-text dataset there are", len(self.documents), "textual documents")
        file.close()
        
        #抽取问题
        filename = '../dataset/test/query-text'
        file = open(filename)
        contents = file.read()
        #print(contents2)
        self.que_documents = re.split('\n/\n',contents[:-3])
        for idx,que_item in enumerate(self.que_documents):
            index_doc = re.split('\n',que_item)
            index = index_doc[0]#re.search('\d+',re.findall('\d+\n',que_item)[0]).group(0)
            self.que_documents[idx] = index_doc[1]
            self.que_index.append(index)
        #print("In the query-text dataset there are ",len(self.que_documents),"query documents")
        file.close()   
        
        #抽取验证结果
        self.rel_index  = [str(i+1) for i in range(len(self.que_index))]
        self.rel_docs = [[] for i in range(len(self.que_index))]
        filename_re = '../dataset/test/rlv-ass'
        file_re = open(filename_re)
        contents_re = file_re.read()
        self.rel_docs = re.split('\n   /\n',contents_re[:-6])
        
        for idx,que_item in enumerate(self.rel_docs):
            que_item = que_item.replace('\n',' ')
            index_doc = que_item.split()
            self.rel_docs[int(index_doc[0])-1] =  index_doc[1:];
        #print("In the rlv-ass dataset there are ",len(self.rel_docs),"answers")
        file_re.close()
    
    #MEDLINE数据集
    def fetch_med_data(self):
        
        #抽取文档
        filename = '../dataset/raw/MED.ALL'
        file = open(filename)
        contents = file.read()
        self.documents = re.split('.I ',contents[2:])
        for idx,doc_item in enumerate(self.documents):
            index_doc = re.split('\n.W\n',doc_item)
            self.file_index.append(index_doc[0])
            index_doc[1] = index_doc[1].replace('-','')
            self.documents[idx] = index_doc[1].replace('\n',' ').strip()
        #print ("In the MED.ALL dataset there are", len(self.documents), "textual documents")
        file.close()
        
        #问题提取
        filename = '../dataset/test/MED.QRY'
        file = open(filename)
        contents = file.read()
        self.que_documents = re.split('.I ',contents[3:])
        for idx,que_item in enumerate(self.que_documents):
            index_doc = re.split('\n.W\n ',que_item)
            #re.search('\d+',re.findall('\d+\n',que_item)[0]).group(0)
            self.que_documents[idx] = index_doc[1].replace('\n',' ').strip()
            self.que_index.append(index_doc[0])
        file.close()
        
        #抽取验证结果
        self.rel_index  = [str(i+1) for i in range(len(self.que_index))]
        self.rel_docs = [[] for i in range(len(self.que_index))]
        filename_re = '../dataset/test/MED.REL'
        file_re = open(filename_re)
        contents_re = file_re.read()
        deal_docs = re.split('\n',contents_re[:-1])
        
        for idx,que_item in enumerate(deal_docs):
            index_doc = que_item.split(" ")
            self.rel_docs[int(index_doc[0])-1].append(index_doc[2]);
        file_re.close()
        
    #LISA数据集
    def fetch_LISA_data(self):
        
        #抽取文档
        filename = '../dataset/raw/LISA0.000'
        file = open(filename)
        contents = file.read()
        self.documents = re.split('\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*',contents)[:-1]
        doc_set = set()
        for idx,doc_item in enumerate(self.documents):
            index_doc = doc_item.split('Document ',1)
            deal_doc = index_doc[1].strip().replace('\n',' ').split(' ',1)
            llen = len(doc_set)
            doc_set.add(deal_doc[0])
            nlen = len(doc_set)
            if llen + 1 == nlen:
                self.file_index.append(deal_doc[0])
                self.documents[idx] = deal_doc[1].strip()
        file.close()
        #print ("In the MED.ALL dataset there are", len(documents2), "textual documents")
        
        
        #问题提取
        filename = '../dataset/test/LISA.QUE'
        file = open(filename)
        contents = file.read()
        self.que_documents = re.split('#\n',contents)[:-1]
        #print ("In the MED.ALL dataset there are", len(que_documents3), "textual documents")
        #print(que_documents3)
        
        for idx,que_item in enumerate(self.que_documents):
            index_doc = que_item.strip().replace('\n',' ').split(' ',1)
            #print(index_doc[0])
            #re.search('\d+',re.findall('\d+\n',que_item)[0]).group(0)
            self.que_documents[idx] = index_doc[1].strip()
            self.que_index.append(index_doc[0])
            
        file.close()
        
        
        
        #答案提取
        
        filename_re = '../dataset/test/LISA.REL'
        file_re = open(filename_re)
        contents_re = file_re.read()
        self.rel_index  = [str(i+1) for i in range(len(self.que_index))]
        self.rel_docs = [[] for i in range(len(self.que_index))]
        rel_documents = re.split('-1',contents_re)[:-1]
        for idx,que_item in enumerate(rel_documents):
            index_doc = re.split('\n\d+ Relevant Refs:\n',que_item)
            #index_doc = que_item.split(" ")
            
            self.rel_docs[idx] = index_doc[1].strip().replace('\n',' ').split(' ');
        file_re.close()
       
        
        
        

        
    
        

