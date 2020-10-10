# -*- coding: utf-8 -*-
import numpy as np


class Evalue(object):
    def __init__(self,doc_sum):
        
        self.lstep = 0
        if doc_sum <= 50:
            self.lstep = 5
        else:
            self.lstep = 10
            
            
        self.plv = []
        self.rlv = []
        
    
    def emrr(self,score_list,que_index,que_docs,rel_docs):
        self.plv = []
        self.rlv = []
        print ("*************开始输出MRR相关结果*************")
        
        #for k in range(self.lstep,len(que_docs)+1,self.lstep):
        for k in range(1,len(que_docs)+1,1):
            
            MRRS = 0
            MRR = 0
            
            for idx,qu_index in enumerate(que_index[:k]):
                Recovery = []
                Precision = []
                qu_index = idx
                #print(len(score_list),len(que_docs)+1,len(score_list[idx]),k)
                find_flag = 0
            
                K = 30
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        find_flag = 1
                        MRRS +=  1/(i + 1.0)
                        break
                    
                        
                if find_flag == 0:
                    MRRS +=  1/30
                    
            MRR = MRRS/k
            print("MRR指标:",MRR)
            print ("***********前"+str(k)+"************")
                        
           
            
            
            
            
            
    def eva_mAP(self,score_list,que_index,que_docs,rel_docs,max_len):
        #print([len(slist)  for slist in score_list])
        #max_len = max([len(slist)  for slist in score_list])
        apy_list = []
        apx_list = []
        p_list = []
        r_list = []
        #print(max_len)
        for idx,qu_index in enumerate(que_index):
            Recovery = []
            Precision = []
            qu_index = idx
            #print(len(score_list),len(que_docs)+1,len(score_list[idx]),k)
            
            have = 0
            for K in range(len(score_list[idx])):
                have = 0
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/((K+1)*1.0))
                    
            
            fill_r = have*1.0/float(len(rel_docs[qu_index]))
            
            
            for K in range(len(score_list[idx]),max_len,1):
                Recovery.append(fill_r)
                Precision.append(have*1.0/((K+1)*1.0))
            Recovery = Recovery[0:max_len]
            Precision = Precision[0:max_len]
            p_list.append(Precision)
            r_list.append(Recovery)
            
            #print(len(Recovery))
        
        
        
        Recovery_avg = (np.array(r_list).sum(axis=0))/len(r_list)
        Recovery_avg = np.around(Recovery_avg,3)
        
        Precision_avg = (np.array(p_list).sum(axis=0))/len(p_list)
        Precision_avg = np.around(Precision_avg,3)
        #print(Precision_avg)
        #print(Recovery_avg)
        
        
        
        apy_list.append(Precision_avg[1] - 0.01)
        apx_list.append(0.0)
        gap = 0.04
        up = gap
         
        for i in range(max_len):
            while Recovery_avg[i] > up:
                apx_list.append(Recovery_avg[i])
                apy_list.append(Precision_avg[i])
                up += gap
        MAP = sum(apy_list)/len(apy_list)
        print("MAP:",MAP)
        for a in range(int(up*100),101,5):
            apx_list.append(a*1.0/100)
            apy_list.append(0.0)
        #print(apy_list)
        #print(apx_list)
        return apy_list,apx_list
        
        
        
        
    
    
    def eva(self,score_list,que_index,que_docs,rel_docs):
        self.plv = []
        self.rlv = []
        print ("*************开始输出PR相关结果*************")
        
        #for k in range(self.lstep,len(que_docs)+1,self.lstep):
        for k in range(1,len(que_docs)+1,1):
            Recovery_list = []
            Precision_list = []
        
            for idx,qu_index in enumerate(que_index[:k]):
                Recovery = []
                Precision = []
                qu_index = idx
                #print(len(score_list),len(que_docs)+1,len(score_list[idx]),k)
            
                have = 0
                K = 5
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                if len(rel_docs[qu_index]) == 0:
                    Recovery.append(0.0)
                    Precision.append(0.0)
                else:   
                    Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                    Precision.append(have*1.0/(K*1.0))
            
                have = 0
                K = 10
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
            
                have = 0
                K = 20
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
                
                have = 0
                K = 30
                #K = 100
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
            
                Recovery_list.append(Recovery)
                Precision_list.append(Precision)
            
            #print("Recovery:\n",Recovery_list)
            #print("Precision:\n",Precision_list)
            Recovery_avg = (np.array(Recovery_list).sum(axis=0))/len(Recovery_list)
            Recovery_avg = np.around(Recovery_avg,3)
            
            Precision_avg = (np.array(Precision_list).sum(axis=0))/len(Precision_list)
            Precision_avg = np.around(Precision_avg,3)
            self.rlv.append(Recovery_avg)
            self.plv.append(Precision_avg)
            print("Recovery:",Recovery_avg,"\nPrecision:",Precision_avg)
        
        
            print ("***********前"+str(k)+"************")
            print ("***************************")
        return Precision_list,Recovery_list
        #print(self.plv)
        #print(self.rlv)
        
            
        
        
        
    def eva_sni(self,score_list,que_index,que_docs,rel_docs):
        self.plv = []
        self.rlv = []
        print ("*************开始输出PR相关结果*************")
        
        
        sni_score_list = []
        record_index = []
        for rel_index,score_list_item in enumerate(score_list):
            temp = score_list_item
            remove = 10
            index_list = []
            #print(len(temp))
            for idx,score in enumerate(temp):
                if idx>= remove:
                    remove += 5
                    
                if (score[0] not in rel_docs[rel_index]) and (idx < remove) and (idx >= remove-5):
                    del score_list_item[idx]
                    remove += 5
                else:
                    index_list.append(score[0])
            #print(len(score_list_item)) 
            sni_score_list.append(score_list_item)
            record_index.append(index_list)

            
        
        score_list = sni_score_list
        #for k in range(self.lstep,len(que_docs)+1,self.lstep):
        for k in range(1,len(que_docs)+1,1):
            Recovery_list = []
            Precision_list = []
            
            
            
            
            
        
            for idx,qu_index in enumerate(que_index[:k]):
                Recovery = []
                Precision = []
                qu_index = idx
                #print(len(score_list),len(que_docs)+1,len(score_list[idx]),k)
            
                have = 0
                K = 5
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
            
                have = 0
                K = 10
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
            
                have = 0
                K = 20
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
                
                have = 0
                K = 30
                for i in range(K):
                    if score_list[idx][i][0] in rel_docs[idx]:
                        have +=1
                Recovery.append(have*1.0/float(len(rel_docs[qu_index])))
                Precision.append(have*1.0/(K*1.0))
            
                Recovery_list.append(Recovery)
                Precision_list.append(Precision)
            
            #print("Recovery:\n",Recovery_list)
            #print("Precision:\n",Precision_list)
            Recovery_avg = (np.array(Recovery_list).sum(axis=0))/len(Recovery_list)
            Recovery_avg = np.around(Recovery_avg,3)
            
            Precision_avg = (np.array(Precision_list).sum(axis=0))/len(Precision_list)
            Precision_avg = np.around(Precision_avg,3)
            self.rlv.append(Recovery_avg)
            self.plv.append(Precision_avg)
            print("Recovery:",Recovery_avg,"\nPrecision:",Precision_avg)
        
        
            print ("***********前"+str(k)+"************")
            print ("***************************")
        return Precision_list,Recovery_list
        #print(self.plv)
        #print(self.rlv)



