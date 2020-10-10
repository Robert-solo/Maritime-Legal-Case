#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:48:29 2019

@author: Robert jun
"""



import fileDeal as fd

#使用何种数据
#c_data = "npl_data"
c_data = "med_data"
#c_data = "LISA_data"

inv_filename = "../support/tr_inv_" + c_data + ".txt"
        
#获取倒排索引库
inverted_word = {}
for line in open(inv_filename,'r'): 
    deal_str = line[:-1].split(":")
    doc_index_list = eval(deal_str[1])
    for word in deal_str[0].split(" "):
        locations = inverted_word.setdefault(word, [])
        locations.extend(doc_index_list)
        
print(inverted_word)
        
docs_have = []  #存储所有查询对应的文档编号集合
test_word = "vertebrate"
if test_word in inverted_word:
    print(inverted_word[test_word])
else:
    print("not found")
                    
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
query:
hemophilia and christmas disease, especially in regard to the
specific complication of pseudotumor formation (occurrence,
pathogenesis, treatment, prognosis).

'''

    
'''


vertebrate host:[726]
host:[193, 195, 481, 531, 659, 976]



hemophilia and christmas disease, especially in regard to the
specific complication of pseudotumor formation (occurrence,
pathogenesis, treatment, prognosis).

30 0 823 1
30 0 825 1
30 0 827 1
30 0 831 1
30 0 843 1
30 0 1019 1
30 0 1020 1
30 0 1021 1
30 0 1022 1
30 0 1024 1
30 0 1026 1
30 0 1027 1
30 0 1032 1
30 0 1033 1


                             

hemophilic:[827, 840, 1022, 1025, 1031, 1032]
pseudotumor: [1022]  [1018, 1025]


hemophilic:[1032]
hemophilic pseudotumor:[1022,1025]
hemophilia:[1026]
angiohemophilia: [1027]

'''
