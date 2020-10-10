# -*- coding: utf-8 -*-
import math
from tqdm import tqdm
from six import iteritems
import numpy as np
from nltk.metrics.distance import jaccard_distance
import wordsim
from nltk.corpus import wordnet as wn

#
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25
K2=1
K3=1



#常用短语提取
class Match(object):
    
    def __init__(self, corpus, w2v, WMD_have = 0):
        self.corpus_size = len(corpus) 
        self.words_size = sum(float(len(x)) for x in corpus)
        self.avgdl = self.words_size / self.corpus_size  
        self.corpus = corpus  
        
        self.f = []
        self.g_f = {}
        self.df = {}
        self.idf = {}
        self.bmidf = {}
        self.g_position = {}
        self.doc_len = []
        self.doc_scope = []
        self.average_idf = 0
        self.word2vec = []
        if WMD_have == 1:
            self.word2vec = w2v
            
        self.simf = wordsim.SimSupport()
        self.idfmax = math.log(self.corpus_size - 1 + 0.5) - math.log(1 + 0.5)
        self.initialize()
        
        #后补充
        self.map_dict = {}
        
    #主要初始化tf-idf
    def initialize(self):
        for index,document in enumerate(self.corpus):
            
            frequencies = {}
            #position = {}
            self.doc_len.append(len(document))
            self.doc_scope.append(len(set(document))-1)
            for pos,word in enumerate(document):
                #print(pos,word)
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
                if word not in self.g_f:
                    self.g_f[word] = 0
                self.g_f[word] += 1
                if word not in self.g_position:
                    self.g_position[word] = set([index])
                self.g_position[word].add(index)
                
                
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] =  math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.bmidf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        
        self.average_idf = sum(map(lambda k: float(self.bmidf[k]), self.bmidf.keys())) / len(self.bmidf.keys())
        print("word sizes:",len(self.bmidf))
        #print(self.bmidf)
    
    
    
    
    
    
    #wordSim1 Lev-W2v 矩阵融合
    def wordSim1(self,sni1,sni2,limit_ratio,filter_score):
        #print(sni1,"\n",sni2)
        sim_score_snippets = []
        doc_sni_length = len(sni2)
        que_sni_length = len(sni1)
        relev_sni_length = 0
        
        
        
        for index,sni_item2 in enumerate(sni2):
            sni_score_list = []
            sni_flag = 0
            nsum = 0
            for sni_item1 in sni1:
                lena = len(sni_item1)
                lenb = len(sni_item2)
                matrixC = np.ones((lena,lenb),dtype = float) 
                matrixIDF = np.ones((lena,lenb),dtype = float)
                for i in range(lena):
                    for j in range(lenb):
                        word1 = sni_item1[i]
                        word2 = sni_item2[j]
                        
                        vec1 = np.array(self.word2vec.get_vector(word1))
                        vec2 = np.array(self.word2vec.get_vector(word2))
                        #matrixC[i,j] = self.simf.eidtString(word1,word2) * self.simf.cosVector(vec1,vec2)
                        matrixC[i,j] = self.simf.cosVector(vec1,vec2)
                        try:
                            matrixIDF[i,j] = (self.bmidf[word1] + self.bmidf[word2])/(2*self.idfmax)
                        except KeyError:
                            matrixIDF[i,j] = 0
                        
                        #matrixC[i,j] = self.simf.eidtString(word1,word2)
                #print(matrixC)
                '''
                sim_sum = 0.
                iter_time = 0
                while lena > 0 and lenb > 0:
                    matrixC,lena,lenb,sim_sum = self.simf.fetchmax(matrixC,lena,lenb,sim_sum)
                    iter_time += 1
                '''
                
                sim_score = self.simf.fetchmax1(matrixC,matrixIDF,lena,lenb)
                sni_score_list.append(sim_score)
                nsum += sim_score
                #当与问句中的一个超过阈值就视为相关
                if sim_score > filter_score:
                    sni_flag = 1
            #评分根据长度 最后取平均
            if sni_flag == 1:
                relev_sni_length += 1
                sim_score_snippets.append((index,nsum/que_sni_length))
            #print(sni_score_list)
            
        rel_snippet_ratio = relev_sni_length*1.0/(doc_sni_length)
        rel_snippet_ratio = rel_snippet_ratio if rel_snippet_ratio < limit_ratio else limit_ratio
        return sim_score_snippets,rel_snippet_ratio
    
    
    
    
    #def simbm25(qu,all_doc_slices[doc_index],1.0):
        
     
    
    
    #bm25公式
    def bmscore(self, que ,doc, slice_index ,index, average_idf,avgdl):
        #avgdl = 20
        score = 0
        for word in que:
            if word not in doc:
                continue
            #print(word)
            idf = self.bmidf[word] if self.bmidf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(doc) / self.avgdl))) # self.avgdl
        return score
    
    
    #获取bm25  选取最大
    def bm25Sim(self, qu_slice, doc_slice, doc_index,limit_ratio,filter_score):
        
        sim_score_snippets = []
        doc_sni_length = len(doc_slice)
        avgdl = sum([len(doc) for doc in doc_slice])/doc_sni_length
        que_sni_length = len(qu_slice)
        relev_sni_length = 0
        
        for slice_index,sni_item2 in enumerate(doc_slice):
            sni_select_flag = 0
            nmax = 0
            for sni_item1 in qu_slice:
                bm_score = self.bmscore(sni_item1,sni_item2,slice_index,doc_index, self.average_idf,avgdl) #*self.Swei(sni_item1,sni_item2)
                if bm_score >= filter_score and bm_score > nmax:
                    sni_select_flag = 1
                    nmax += bm_score
                    #nmax = bm_score
                    
            if sni_select_flag == 1:
                sim_score_snippets.append((slice_index,nmax/que_sni_length)) #nmax
                relev_sni_length += 1
        
        rel_snippet_ratio = relev_sni_length*1.0/(doc_sni_length)
        rel_snippet_ratio = rel_snippet_ratio if rel_snippet_ratio < limit_ratio else limit_ratio
        #print(rel_snippet_ratio)
        return sim_score_snippets,rel_snippet_ratio
    
    
    #计算句子权重
    def Swei(self, sni_item1,sni_item2):
        n = 0
        wei = 0.01
        for word2 in sni_item2:
            for word1 in sni_item1:
               if word1 == word2:
                   n+=1
                   wei += self.bmidf[word2]
                   break
        #print(wei/max(n,0.1))
        wei = wei/ (max(n,0.1) * self.idfmax)
        
        return  wei# max(wei, self.average_idf)
    
    
    #获取WMD  成本装换代价
    def wmdCost(self, qu_slice, doc_slice,limit_ratio,filter_score):
        
        sim_score_snippets = []
        doc_sni_length = len(doc_slice)
        relev_sni_length = 0
        
        
        for slice_index,sni_item2 in enumerate(doc_slice):
            sni_select_flag = 0
            nmax = 0
            '''
            wei = 0
            idf_list = []
            for sni_word in sni_item2:
                idf_list.append(self.bmidf[sni_word])
            idf_list.sort(reverse=True)
            
            if  len(idf_list) >= 3:
                wei = sum(idf_list[:3])/3
            else:
                wei = sum(idf_list)/len(idf_list)
            '''
            
            for sni_item1 in qu_slice:
                wmd_score =  self.Swei(sni_item1,sni_item2) * 100 / (self.word2vec.wmdistance(sni_item1,sni_item2)+0.01) 
                #print(wmd_score)
                
                
                if wmd_score >= filter_score and wmd_score > nmax:
                    #print(wmd_score)
                    sni_select_flag = 1
                    nmax =  wmd_score
                    
                    
                    
            if sni_select_flag == 1:
                sim_score_snippets.append((slice_index, nmax))
                relev_sni_length += 1
        
        rel_snippet_ratio = relev_sni_length*1.0/(doc_sni_length)
        rel_snippet_ratio = rel_snippet_ratio if rel_snippet_ratio < limit_ratio else limit_ratio
        #print(rel_snippet_ratio)
        return sim_score_snippets,rel_snippet_ratio
        
    #求lcs
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



    #计算lcs + lcs + jaccard 分数
    def compute_lcs(self,doc1,doc2,):
        LCS,MCLCS1,MCLCSN = self.lcs(doc1,doc2)
        jaccard_score = 1 - jaccard_distance(set(doc1),set(doc2))
        #score =  (1.0 * LCS + 1.0 * MCLCSN +  1.0 * jaccard_score)/3
        score =  0.1 * LCS  + 0.3 * MCLCSN + 0.6 * jaccard_score
        #score = jaccard_score
        return score
    
    #传统方法加权求和
    def lmjSim(self, qu_slice, doc_slice, limit_ratio, filter_score):
        
        sim_score_snippets = []
        doc_sni_length = len(doc_slice)
        relev_sni_length = 0
        
        for slice_index,sni_item2 in enumerate(doc_slice):
            sni_select_flag = 0
            nmax = 0
            for sni_item1 in qu_slice:
                lmj_score = self.compute_lcs(sni_item1,sni_item2) * self.Swei(sni_item1,sni_item2)
                #print(lmj_score)
                if lmj_score >= filter_score and lmj_score > nmax:
                    sni_select_flag = 1
                    nmax = lmj_score 
                    
                    
            if sni_select_flag == 1:
                sim_score_snippets.append((slice_index,nmax))
                relev_sni_length += 1
        
        rel_snippet_ratio = relev_sni_length*1.0/(doc_sni_length)
        rel_snippet_ratio = rel_snippet_ratio if rel_snippet_ratio < limit_ratio else limit_ratio
        #print(rel_snippet_ratio)
        return sim_score_snippets,rel_snippet_ratio
    
    
    
    #单词级  相关性累和
    def getwordsim1(self,all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,ing_fun):
        
        score_list = []

        pbar = tqdm(total= len(all_que_slices))
        update_step = 100/len(all_que_slices)

        for idx,qu in enumerate(all_que_slices):
            pbar.update(update_step)
            wordsim1_score_list = []
            doc_have =  docs_have[idx]
            
            for doc_index in doc_have:
                sim_score_snippets, rel_snippet_ratio = self.wordSim1(qu,all_doc_slices[doc_index],0.6,0.3)
                #print(sim_score_snippets, rel_snippet_ratio)
                score = self.integScore(sim_score_snippets, rel_snippet_ratio,ing_fun)
                wordsim1_score_list.append((str(file_index[doc_index]),score))
                print(idx,str(file_index[doc_index]),score)
            
            wordsim1_score_list.sort(key=lambda x:x[1],reverse=True)
            score_list.append(wordsim1_score_list[:50])
        pbar.close()
        return  score_list
    
    
    
    #句子级相关性  wmd
    def getsimbm25(self,all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,ing_fun):
        
        score_list = []
        for idx,qu in enumerate(all_que_slices):
            wordsim1_score_list = []
            doc_have = docs_have[idx]
            
            for doc_index in doc_have:
                #print(doc_index)
                sim_score_snippets, rel_snippet_ratio = self.bm25Sim(qu,all_doc_slices[doc_index],doc_index, 0.45, self.average_idf+1) #self.average_idf
                #sim_score_snippets, rel_snippet_ratio = self.bm25Sim(qu,all_doc_slices[doc_index],doc_index, 0.35, self.average_idf+1) #self.average_idf
                '''
                if idx == 2 and (doc_index == 406 or  doc_index == 905 or doc_index == 61 or  doc_index == 229):
                    print("doc_index:", doc_index+1, "sim_score_snippets:", sim_score_snippets, "rel_snippet_ratio:", rel_snippet_ratio )
                '''
                '''
                if idx == 4 and (doc_index == 1309 or  doc_index == 5795 or doc_index == 3599 or  doc_index == 3809):
                    print("doc_index:", file_index[doc_index], "sim_score_snippets:", sim_score_snippets, "rel_snippet_ratio:", rel_snippet_ratio )
                '''
                '''
                if idx == 5 and (doc_index == 663 or  doc_index == 2420 or doc_index == 3404 or  doc_index == 5920):
                    print("doc_index:", file_index[doc_index], "sim_score_snippets:", sim_score_snippets, "rel_snippet_ratio:", rel_snippet_ratio )
                
                '''
                '''
                正
                3410 2426
                
                错入:
                664 5926
                '''
                #print(sim_score_snippets, rel_snippet_ratio)
                score = self.integScore(sim_score_snippets, rel_snippet_ratio,ing_fun)
                wordsim1_score_list.append((str(file_index[doc_index]),score))
                
                #print(idx,str(file_index[doc_index]),score)
            
            
            wordsim1_score_list.sort(key=lambda x:x[1],reverse=True)
            score_list.append(wordsim1_score_list)
        
        return  score_list
    
    
    
    
    
    
    
    #句子级相关性  wmd
    def getsimbm25forfullsearch(self,all_doc_slices,all_que_slices,file_index,ing_fun):
        
        score_list = []
        #score_dict = {}
        for qu_idx,qu in enumerate(all_que_slices):
            bm25sim_score_list = []
            for idx,doc_slice_item in enumerate(all_doc_slices):
                sim_score_snippets, rel_snippet_ratio = self.bm25Sim(qu,doc_slice_item,idx, 0.45, self.average_idf + 1)
                score = self.integScore(sim_score_snippets, rel_snippet_ratio,ing_fun)
                bm25sim_score_list.append((str(file_index[idx]),score))
            
            bm25sim_score_list.sort(key=lambda x:x[1],reverse=True)
            score_list.append(bm25sim_score_list[:300])
            
        return  score_list
        
    
    
    
    
    
    
    #句子级相关性  wmd
    def getsimwmd(self,all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,ing_fun):
        
        score_list = []
        
        for idx,qu in enumerate(all_que_slices):
            wordsim1_score_list = []
            doc_have = docs_have[idx]
            
            for doc_index in doc_have:
                sim_score_snippets, rel_snippet_ratio = self.wmdCost(qu,all_doc_slices[doc_index],0.55,3.5)
                #print(sim_score_snippets, rel_snippet_ratio)
                score = self.integScore(sim_score_snippets, rel_snippet_ratio,ing_fun)
                wordsim1_score_list.append((str(file_index[doc_index]),score))
                #print(idx,str(file_index[doc_index]),score)
            
            wordsim1_score_list.sort(key=lambda x:x[1],reverse=True)
            score_list.append(wordsim1_score_list[:50])
        
        return  score_list
    
    
    
    
    
    #句子级 lcs+mclcs+jaccard
    def getsimlmj(self,all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,ing_fun):
        score_list = []
        
        for idx,qu in enumerate(all_que_slices):
            wordsim1_score_list = []
            doc_have = docs_have[idx]
            
            for doc_index in doc_have:
                sim_score_snippets, rel_snippet_ratio = self.lmjSim(qu,all_doc_slices[doc_index],1.0,0.0)
                #print(sim_score_snippets, rel_snippet_ratio)
                score = self.integScore(sim_score_snippets, rel_snippet_ratio,ing_fun)
                wordsim1_score_list.append((str(file_index[doc_index]),score))
                print(idx,str(file_index[doc_index]),score)
            
            wordsim1_score_list.sort(key=lambda x:x[1],reverse=True)
            score_list.append(wordsim1_score_list[:50])
        
        return  score_list
        

          
    



        
    def test2(self,all_doc_slices,all_que_slices,rel_docs,docs_have):
        doc_sum = self.corpus_size
        
        qu = all_que_slices[0]
        #doc_have = docs_have[0][0:2]
        doc_have =  [ int(ind)-1 for ind in rel_docs[0][0:4]]
        wordsim1_score_list = [0] * doc_sum
        
        for doc_index in doc_have:
            sim_score_snippets, rel_snippet_ratio = self.test1(qu,all_doc_slices[doc_index],1.0)
            print(sim_score_snippets, rel_snippet_ratio)
            
        
        return  wordsim1_score_list
    
    
    def ewGen_occ(self,que_doc,corpus_words,all_doc_slices,minconf):
        
        
        exword_list = []
        for qu_word in que_doc:
            if (qu_word not in self.bmidf) or (self.bmidf[qu_word] <= self.average_idf):
                print(qu_word +" pass")
                continue
            for word in corpus_words:
                if (word in que_doc) or (word not in self.bmidf) or (self.bmidf[word] <= self.average_idf ):
                    continue
                #ex_flag = 0
                #查询项,拓展项,共现次数
                occ_qu = 0
                occ_ex = 0
                occ_oc = 0
                for doc_slices in all_doc_slices:
                    for doc_slice in doc_slices:
                        if qu_word in doc_slice:
                            occ_qu += 1
                        if word in doc_slice:
                            occ_ex += 1
                        if qu_word in doc_slice and word in doc_slice:
                            occ_oc += 1
                #conf = occ_oc*1.0/(occ_qu + occ_ex - occ_oc - 0.000000001)
                conf = occ_oc*1.0/(occ_qu  - 0.000000001)
                #conf = occ_oc*1.0/( occ_ex - 0.000000001)
                if conf > minconf:
                    print(qu_word," - ",word,":",conf ,"..", occ_qu," ",occ_ex," ",occ_oc)
                    exword_list.append(word)
                    locations = self.map_dict.setdefault(qu_word, []) 
                    locations.append(word) 
        print(exword_list)
        que_doc.extend(exword_list)
        que_doc = list(set(que_doc))
        return que_doc
                
                
    
    def ewGen_wn(self,que_doc):
        exword_list = []
        temp_list = []
        for qu_word in que_doc:
            ss = wn.synsets(qu_word)
            for synset in ss:
                temp_list.extend(synset.lemma_names())
            
        
        temp_list = list(set(temp_list))
        for ex_word in temp_list:
            if ex_word not in self.bmidf:
                continue
            if self.bmidf[ex_word] >self.average_idf:
                exword_list.append(ex_word)
                print(ex_word,"的idf值: ",self.bmidf[ex_word])
            
        que_doc.extend(exword_list)
        que_doc = list(set(que_doc))
        return que_doc


    def ewGen_senti(self,que_docs):
        
        
        
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #分数整合
    def integScore(self,scores, score_ratio,ing_fun):
        res_score = 0
        snippet_score_list = [score[1] for score in scores]
        
        
        if len(snippet_score_list) == 0:
            return res_score
        snippet_score_list.sort(reverse=True) 
       
        if len(snippet_score_list) >= 5 :
            snippet_score_list = snippet_score_list[:5]
        
        if ing_fun == 1:
            res_score = ( K3 * np.max(snippet_score_list) + np.average(snippet_score_list)) * score_ratio
        elif ing_fun == 2:
            res_score = K3 * np.max(snippet_score_list) * score_ratio
        elif ing_fun == 3:
            rlen = len(snippet_score_list)
            weigth_whole = rlen*(rlen+1)/2
            
            s1 = 0
            for i in range(rlen):
                s1+= ((rlen-i)/weigth_whole * snippet_score_list[i])
            res_score = s1 * score_ratio
        
        return res_score
    
    
    
    
    
    
    
    
        '''
        for i in range(self.kqu):
            
            just_wmd_score_list = [0] * self.doc_sum
            wmd_score_flags = [1] * self.doc_sum
            tmp_wmd_score_list = []
            
            for slice_pos in range(split_fixed_length):
                #阶段一,切片
                all_doc_slices = self.docSnippetModel.get_doc_block_for_slide(split_fixed_length,slice_pos)
                
                for index in range(len(self.docs_have[i])):
                    doc_index  = int(self.docs_have[i][index])
                    
                    if wmd_score_flags[doc_index] == 1:
                        #阶段二,筛片
                        wmd_score_snippets, rel_snippet_ratio = self.multiScoreModel.get_wmd_scores(self.que_docs[i],
                                                                                                    all_doc_slices[doc_index],
                                                                                                    self.limit_ratio,
                                                                                                    self.filter_score)
                        #print(rel_snippet_ratio)
                        if len(wmd_score_snippets) <= 1:
                            wmd_score_flags[doc_index] = 0
                        #阶段三,整和选择
                        wmd_score = self.docSnippetModel.integScore(wmd_score_snippets, rel_snippet_ratio,self.ing_fun)
                        if wmd_score > just_wmd_score_list[doc_index]:
                            just_wmd_score_list[doc_index] = wmd_score
            
            for index in range(len(self.docs_have[i])):
                doc_index = self.docs_have[i][index]
                try:
                    tmp_wmd_score_list.append((str(self.file_index[doc_index]),just_wmd_score_list[doc_index]))
                except:
                    tmp_wmd_score_list.append(('0',100.0))
                    e_t += 1
                    print("e_t:",e_t)
            tmp_wmd_score_list.sort(key=lambda x:x[1],reverse=True)
            self.sni_score_list.append(tmp_wmd_score_list[:50])
        '''
    

            
            
    
        
    
        
        
        
    
    
        
        
        

        
        
        