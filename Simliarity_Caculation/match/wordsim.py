#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np


class SimSupport(object):
    def __init__(self,):
       pass
        
    #字符串距离      
    def eidtString(self,s1, s2):
        # 矩阵的下标得多一个
        len_str1 = len(s1) + 1
        len_str2 = len(s2) + 1
     
        # 初始化了一半  剩下一半在下面初始化
        matrix = [[0] * (len_str2) for i in range(len_str1)]
     
        for i in range(len_str1):
            for j in range(len_str2):
                if i == 0 and j == 0:
                    matrix[i][j] = 0
                # 初始化矩阵
                elif i == 0 and j > 0:
                    matrix[0][j] = j
                elif i > 0 and j == 0:
                    matrix[i][0] = i
                # flag
                elif s1[i - 1] == s2[j - 1]:
                    matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
                else:
                    matrix[i][j] = min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
        return max(1 - matrix[len_str1 - 1][len_str2 - 1] * 1.0 / max(len(s1),len(s2)) , 0.2)



    #余弦相似度
    def cosVector(self,x,y):
        if(len(x)!=len(y)):
            print('error input,x and y is not in the same space')
            return;
        result1=0.0;
        result2=0.0;
        result3=0.0;
        for i in range(len(x)):
            result1+=x[i]*y[i]   #sum(X*Y)
            result2+=x[i]**2     #sum(X*X)
            result3+=y[i]**2     #sum(Y*Y)
        #print(result1)
        #print(result2)
        #print(result3)
        cosv = result1/((result2*result3 + 0.001)**0.5)
        return cosv
    
    
    #水平垂直相似度
    def weightRo(self,x,y):
        if(len(x)!=len(y)):
            print('error input,x and y is not in the same space')
            return;
        result1=0.0;
        result2=0.0;
        result3=0.0;
        result4=0.0;
        result5=0.0;
        
        for i in range(len(x)):
            result1+=x[i]*y[i]   #sum(X*Y)
            result2+=x[i]**2     #sum(X*X)
            result3+=y[i]**2     #sum(Y*Y)
        #print(result1)
        #print(result2)
        #print(result3)
        
        parall = result1/result3 * y
        prepen = x - parall
        
        for i in range(len(x)):
            result4+=parall[i]**2     #sum(X*X)
            result5+=prepen[i]**2     #sum(Y*Y)
        
        ratio = result4/result5
        return ratio
    
    
    

    
    
    #相似度句子整和方案1,矩阵划分
    def fetchmax(self,c,d,lena,lenb):
        sim_sum = 0
        iter_time = 0
        while lena > 0 and lenb > 0 and iter_time < 3:
            max_s = -1 
            pos_line = 0
            pos_col = 0
            for i in range(lena):
                for j in range(lenb):
                    if c[i,j] >= max_s:
                        max_s = c[i,j]
                        pos_line = i
                        pos_col = j
            
            sim_sum += max_s
            lena-=1
            lenb-=1
            iter_time +=1
            c = np.delete(c,pos_line,axis = 0)
            if lena == 0:
                return sim_sum/iter_time
            c = np.delete(c,pos_col,axis = 1)
        return sim_sum/iter_time
    
    
    
    #相似度句子整和方案1,矩阵划分修改版 行列集合遍历式
    def fetchmax1(self,c,d,lena,lenb):
        sim_sum = 0
        iter_time = 0
        seta = set([ k for k in range(lena)])
        setb = set([ k for k in range(lenb)])
        while lena > 0 and lenb > 0 and iter_time < 3:
            max_s = -1 
            pos_line = 0
            pos_col = 0
            #print(seta,setb)
            for i in seta:
                for j in setb:
                    if c[i,j] >= max_s:
                        max_s = c[i,j] 
                        pos_line = i
                        pos_col = j
            
            sim_sum += max_s * d[pos_line , pos_col]
            lena -= 1
            lenb -= 1
            iter_time += 1
            seta.remove(pos_line)
            if lena == 0:
                return sim_sum/iter_time
            setb.remove(pos_col)
        return sim_sum/iter_time
    
    #按行求最大
    def fetchmax2(self,c,lena,lenb):
        sim_sum = 0
        iter_time = 0
        while lena > 0 and lenb > 0:
            max_s = 0 
            pos_line = 0
            pos_col = 0
            for i in range(lena):
                for j in range(lenb):
                    if c[i,j] >= max_s:
                        max_s = c[i,j]
                        pos_line = i
                        pos_col = j
            
            sim_sum += max_s
            lena-=1
            lenb-=1
            iter_time +=1
            c = np.delete(c,pos_line,axis = 0)
            if lena == 0:
                return sim_sum/iter_time
            c = np.delete(c,pos_col,axis = 1)
        return sim_sum/iter_time
    
    
    #相似度句子整和方案2,最大累和均值
    
    
    
    
    