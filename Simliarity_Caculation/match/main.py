# -*- coding: utf-8 -*-




import fileDeal as fd
import process as pp

'''
0.重要参数
'''

#使用何种数据
#c_data = "npl_data"
c_data = "med_data"
#c_data = "LISA_data"


#使用何种方法
#Match_fun = "wordSim1"
#Match_fun = "wordSim2"
#Match_fun = "bm25"
Match_fun = "wmd"
#Match_fun = "lmj"


w2v_kind = 1
word2vec = []
if Match_fun == "wmd" or Match_fun == "wordSim1" or Match_fun == "wordSim2":
    import gensim
    if w2v_kind == 1:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove-wiki-gigaword-100/glove-wiki-gigaword-100.txt',binary=False)
    elif w2v_kind == 2:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/glove.6B/glove.6B.100d.txt',binary=False)
    elif w2v_kind == 3:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/python_workspace/datahouse/word2vec-google-news-300/GoogleNews-vectors-negative300.bin',binary=True)



'''
1.数据提取
'''
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

print("In the doc-text dataset there are", len(documents), "textual documents")
print("In the query-text dataset there are ",len(que_index),"query documents")
print("In the rlv-ass dataset there are ",len(rel_docs),"answers")

print("In the collection words per document are ",perwords)



'''
print(file_index[:3],len(file_index))
print(documents[:3])
print(que_index[:3],len(que_index))
print(que_documents[:3])

print(rel_index,len(rel_index))
print(rel_docs)
'''

'''
2.数据预处理
'''
#all_que_slices  all_doc_slices 为按句子切分的查询候选片段   que_docs processed_docs 为一般化处理的查询候选文档
preprocess = pp.PreProc(que_documents,documents)
all_que_slices = preprocess.Puncut(0)
all_doc_slices = preprocess.Puncut(1)
que_docs,processed_docs = preprocess.Normal()

#print(all_doc_slices[:3])
#print(len(all_que_slices))
#print(all_que_slices[28])

#all_que_slices = all_que_slices[:3]
#que_docs = que_docs[:3]
docs_have = preprocess.FetchDocs(c_data, que_docs,rel_docs)



'''
3.数据碎片化机制
'''
import matchTool

ma = matchTool.Match(processed_docs,word2vec,1)
#print(ma.idfmax)
#sni1_score_list = ma.getwordsim1(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,2)
#sni2_score_list = ma.getwordsim1(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,3)
sni1_score_list = ma.getsimwmd(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,2)
sni2_score_list = ma.getsimwmd(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,3)

#ma = matchTool.Match(processed_docs,[],0)

#针对med数据集
#sni1_score_list = ma.getsimbm25(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,2)
#sni2_score_list = ma.getsimbm25(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,3)


#针对LISA数据集
#sni1_score_list = ma.getsimbm25(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,1)
#sni2_score_list = ma.getsimbm25(all_doc_slices,all_que_slices,rel_docs,docs_have,file_index,3)


'''
3.原始方案
'''

import maOrigin
mo = maOrigin.MatchOrigin(processed_docs,que_docs,file_index,docs_have,word2vec)     #原始方案初始化
mo.getMatch(Match_fun)                                            #开始匹配
score_list = mo.score_list



'''
import maMusnippet
ma = maMusnippet.MatchMusnippet(processed_docs,que_docs,file_index,docs_have,word2vec)

ma.getMatch(Match_fun,filter_score = 1.0,limit_length = 5,limit_ratio = 0.5, ing_fun = 2)
sni1_score_list = ma.sni_score_list
ma.getMatch(Match_fun,filter_score = 1.0,limit_length = 5,limit_ratio = 0.5, ing_fun = 3)
sni2_score_list = ma.sni_score_list
'''



'''
评估结果
'''
import evalue
eva = evalue.Evalue(len(que_docs))

#原始方法评估
#eva.emrr(score_list,que_index,que_docs,rel_docs)
#eva.emrr(sni1_score_list,que_index,que_docs,rel_docs)


pl,rl = eva.eva(score_list,que_index,que_docs,rel_docs)
plv = eva.plv
rlv = eva.rlv
sni1_pl,sni1_rl = eva.eva(sni1_score_list,que_index,que_docs,rel_docs)
sni1_plv = eva.plv
sni1_rlv = eva.rlv

sni2_pl,sni2_rl = eva.eva(sni2_score_list,que_index,que_docs,rel_docs)
sni2_plv = eva.plv
sni2_rlv = eva.rlv


import resultShow 
rs = resultShow.ResultShow()
#print(sni1_score_list[4][:20])
#差距列表
print("差距列表:")
dis_list,top_list = rs.show_dispa(pl,rl,sni1_pl,sni1_rl,0)
print(dis_list)
dis_list,top_list = rs.show_dispa(pl,rl,sni2_pl,sni2_rl,0)
print(dis_list)




rs.load_Result(pl,rl,sni1_pl,sni1_rl,sni2_pl,sni2_rl,1)
import datetime
today = datetime.date.today()
rs.show_F1(1,c_data ,Match_fun,str(today))

#mAP

max_len = max([len(slist)  for slist in sni1_score_list])
pl,rl = eva.eva_mAP(score_list,que_index,que_docs,rel_docs,300)
sni1_pl,sni1_rl = eva.eva_mAP(sni1_score_list,que_index,que_docs,rel_docs,300)
sni2_pl,sni2_rl = eva.eva_mAP(sni2_score_list,que_index,que_docs,rel_docs,300)

print(pl,rl)
print(sni1_pl,sni1_rl)
rs.load_Result(pl,rl,sni1_pl,sni1_rl,sni2_pl,sni2_rl,1)


rs.show_mAP(pl,rl,sni1_pl,sni1_rl,sni2_pl,sni2_rl,c_data,str(today))



'''
rs.load_Result(pl[-1],rl[-1],sni1_pl[-1],sni1_rl[-1],sni2_pl[-1],sni2_rl[-1],1)
rs.show_PR()
'''


'''
dis_list,top_list = rs.show_dispa(pl,rl,sni1_pl,sni1_rl,0)
print(dis_list)

dis_list,top_list = rs.show_dispa(pl,rl,sni2_pl,sni2_rl,0)
print(dis_list)

print(top_list[4])
'''

'''

print(rel_docs[5])
print(sni1_score_list[5][:20])
print(score_list[5][:20])

print(que_index[4],":",que_documents[4],"\n",all_que_slices[4])
print(que_index[5],":",que_documents[5],"\n",all_que_slices[5])

z_d = [1309,2420,3404,5795]
f_d = [663,3599,3809,5920]

for i in z_d:
    print(file_index[i],":",documents[i],"\n",all_doc_slices[i])
    
print("***********************")
for i in f_d:
    print(file_index[i],":",documents[i],"\n",all_doc_slices[i])
'''

