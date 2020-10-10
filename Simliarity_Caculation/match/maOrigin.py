# -*- coding: utf-8 -*-
from gensim.summarization import bm25
from nltk.metrics.distance import jaccard_distance


#常用短语提取
class MatchOrigin(object):
    def __init__(self,processed_docs,que_docs,file_index,docs_have,word2vec):
        
        
        self.processed_docs = processed_docs
        self.que_docs = que_docs
        self.file_index = file_index
        
        self.kqu = len(que_docs)
        self.docs_have = docs_have
        self.word2vec = word2vec
        self.score_list = []
        
        
        
    def getMatch(self,Match_fun):
        if Match_fun == "bm25" :
            self.obm25()
        elif Match_fun == "wmd":
            self.wmd()
        elif Match_fun == "lmj":
            self.lmj()
        else:
            self.wmd()
    
    def obm25(self):
        
        bm25Model = bm25.BM25(self.processed_docs)
        average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        score_dict = {}
        
        for i,que_item in enumerate(self.que_docs):
            scores = bm25Model.get_scores(que_item,average_idf)
            for idx,score_item in enumerate(scores):
                score_dict[self.file_index[idx]] = score_item
            self.score_list.append(sorted(score_dict.items(), key = lambda score_dict:score_dict[1],reverse=True))
        
        '''
        bm25Model = bm25.BM25(self.processed_docs)
        average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
        
        print(len(self.processed_docs))
        ori_docs = self.processed_docs
        for i,que_item in enumerate(self.que_docs):
            score_dict = {}
            now_processed_docs= []
            doc_have = self.docs_have[i]
            for doc_index in doc_have:
                now_processed_docs.append(ori_docs[doc_index])  
            bm25Model = bm25.BM25(now_processed_docs)
            scores = bm25Model.get_scores(que_item,average_idf)
            for idx,score_item in enumerate(scores):
                score_dict[str(self.file_index[doc_have[idx]])] = score_item
            self.score_list.append(sorted(score_dict.items(), key = lambda score_dict:score_dict[1],reverse=True))
        '''
        
    
    
    def wmd(self):
        
        '''
        for i,que_item in enumerate(self.que_docs):
            temp_scores_list = []
            for idx,processed_doc in enumerate(self.processed_docs):
                temp_scores_list.append((self.file_index[idx],self.word2vec.wmdistance(que_item, processed_doc)))
            temp_scores_list = sorted(temp_scores_list, key=lambda x:x[1])[:50]
            self.score_list.append(temp_scores_list)
        '''
        ori_docs = self.processed_docs
        for i,que_item in enumerate(self.que_docs):
            now_processed_docs= []
            doc_have = self.docs_have[i]
            for doc_index in doc_have:
                now_processed_docs.append(ori_docs[doc_index]) 
                
            temp_scores_list = []
            for idx,processed_doc in enumerate(now_processed_docs):
                temp_scores_list.append( (str(self.file_index[doc_have[idx]]), self.word2vec.wmdistance(que_item, processed_doc)) )
            temp_scores_list = sorted(temp_scores_list, key=lambda x:x[1])[:50]
            self.score_list.append(temp_scores_list)
    
    
    
    def lmj(self):
        
        for i,que_item in enumerate(self.que_docs):
            temp_scores_list = []
            for doc_index in self.docs_have[i]:
                try:
                    socre = self.compute_lcs(que_item, self.processed_docs[doc_index])
                    temp_scores_list.append((str(self.file_index[doc_index]), socre))
                except:
                    print(self.docs_have[i])
                    print(self.doc_index)
                
                #print(doc_index+1, socre)
            temp_scores_list = sorted(temp_scores_list, key=lambda x:x[1], reverse=True)[:50]
            self.score_list.append(temp_scores_list)
        
    
    #长文本计算分
    def compute_lcs(self,doc1,doc2):
        LCS,MCLCS1,MCLCSN = self.lcs(doc1,doc2)
        jaccard_score = 1 - jaccard_distance(set(doc1),set(doc2))
        #score =  ( LCS +   MCLCSN + jaccard_score)/3.0
        score =  0.1 * LCS + 0.3 *  MCLCSN + 0.6 * jaccard_score
        #score = jaccard_score
        return score
    
    
    def lcs(self,a,b):
        lena=len(a)
        lenb=len(b)
        c=[[0 for i in range(lenb+1)] for j in range(lena+1)]
        n=[[0 for i in range(lenb+1)] for j in range(lena+1)]
        flag=[[0 for i in range(lenb+1)] for j in range(lena+1)]
        
        LCS = 0
        MCLCS1 = 0
        MCLCSN = 0
        for i in range(lena):
            for j in range(lenb):
                if a[i]==b[j]:
                    c[i+1][j+1]=c[i][j]+1
                    n[i+1][j+1]=n[i][j]+1
                    MCLCSN = max(n[i+1][j+1],MCLCSN)
                    flag[i+1][j+1]='ok'
                elif c[i+1][j]>c[i][j+1]:
                    c[i+1][j+1]=c[i+1][j]
                    n[i+1][j+1] = 0
                    flag[i+1][j+1]='left'
                else:
                    c[i+1][j+1]=c[i][j+1]
                    n[i+1][j+1] = 0
                    flag[i+1][j+1]='up'
        LCS = c[lena][lenb]
        n1 = min(lena,lenb)
        n2 = (lena+lenb)
        for i in range(n1):
            if n[i+1][i+1]==n[i][i]+1:
                MCLCS1 +=1
            else:
                break 
        return LCS/n2,MCLCS1/n2,MCLCSN/n2

