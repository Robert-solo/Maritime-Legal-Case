import gensim
import jieba
import numpy as np
from scipy.linalg import norm
from gensim.models import word2vec

model_file = 'D:\\python\\dengna_src\\law.model'
model =word2vec.Word2Vec.load(model_file)


def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s,  cut_all=False)
        #print("words:", words)
        words=[word  for word in words if word not in "、|，"]
##        words = ' '.join(words).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
##        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
##        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
##        .replace('’', '')     # 去掉标点符号
        #print(words)
        
        v = np.zeros(200)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


#sentence1="抢劫"
sentence1="渔船用5厘米尺寸的渔网捕鱼"
sentence2="使用炸鱼、毒鱼、电鱼等破坏渔业资源方法进行捕捞"
##sentence2=sentence2.replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
##        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
##        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
##        .replace('’', '').replace(' ','')    # 去掉标点符号
sentence3="抢夺公私财物"
sentence4="使用小于最小网目尺寸的网具进行捕捞"


words = jieba.lcut(sentence3,  cut_all=False)

#print(words)
##
##words2=[word  for word in words if word not in "、|，"]
##print(words2)
                                      
        
print(vector_similarity(sentence1,sentence2))
print(vector_similarity(sentence1,sentence3))
print(vector_similarity(sentence1,sentence4))




