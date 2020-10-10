




w2v_kind = 1

import gensim
if w2v_kind == 1:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove-wiki-gigaword-300/glove-wiki-gigaword-300.txt',binary=False)
elif w2v_kind == 2:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove.6B/glove.6B.300d.txt',binary=False)
        
        
#字符串距离      
def eidt_1(s1, s2):
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
def cosVector(x,y):
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
    cosv = result1/((result2*result3)**0.5)
    return cosv




#水平垂直相似度
import numpy as np
def weightRo(x,y):
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


from nltk.corpus import wordnet



word1 = "sleep"
word2 = "sheep"
#print("the cosin result is "+ str(cosVector([1,2,3],[1,2,2])))
#print("the distance result is " + str(eidt_1(word1, word2)))
print(word1,word2)
vec1 = np.array(word2vec.get_vector(word1))
vec2 = np.array(word2vec.get_vector(word2))
c1 = wordnet.synsets(word1)[0] 
c2 = wordnet.synsets(word2)[0] 
print(c1.wup_similarity(c2))
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )





word1 = "shoot"
word2 = "shook"
print(word1,word2)
vec1 = word2vec.get_vector(word1)
vec2 = word2vec.get_vector(word2)
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )

word1 = "surgery"
word2 = "surgical"
print(word1,word2)
vec1 = word2vec.get_vector(word1)
vec2 = word2vec.get_vector(word2)
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )

word1 = "palliate"
word2 = "palliation"
print(word1,word2)
vec1 = word2vec.get_vector(word1)
vec2 = word2vec.get_vector(word2)
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )

word1 = "technology"
word2 = "technique"
print(word1,word2)
vec1 = word2vec.get_vector(word1)
vec2 = word2vec.get_vector(word2)
c1 = wordnet.synsets(word1)[0] 
c2 = wordnet.synsets(word2)[0] 
print(c1.wup_similarity(c2))
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )

word1 = "river"
word2 = "boat"
print(word1,word2)
vec1 = word2vec.get_vector(word1)
vec2 = word2vec.get_vector(word2)
print("the cosin result is "+ str(cosVector(vec1,vec2)))
print("the distance result is " + str(eidt_1(word1, word2)))
print("the ratio result is " + str(weightRo(vec1,vec2)))
print("the acc result is " + str(cosVector(vec1,vec2)*eidt_1(word1, word2)) + "\n" )






c1 = wordnet.synsets(word1)[0] 
c2 = wordnet.synsets(word2)[0] 
print(c1.wup_similarity(c2))


'''
import numpy as np
vec1 = [1,2,3]
vec2 = [1,2,2]
a = np.array(vec1)
b = np.array(vec2)
s = 2 * a

print(list(s))
'''

'''
word1 = "shoot"
word2 = "shook"
word1 = "surgery"
word2 = "surgical"
word1 = "palliate"
word2 = "palliation"
'''










