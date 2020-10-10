# -*- coding: utf-8 -*-
from stanfordcorenlp import StanfordCoreNLP


sentence = "the book is very interesting."

nlp = StanfordCoreNLP(r'/home/leyv/java/stanford-corenlp-full-2018-10-05',lang = 'en')

print ('Tokenize:', nlp.word_tokenize(sentence))
print ('Part of Speech:', nlp.pos_tag(sentence))
print ('Named Entities:', nlp.ner(sentence))
print ('Constituency Parsing:', nlp.parse(sentence))#语法树
print ('Dependency Parsing:', nlp.dependency_parse(sentence))#依存句法
nlp.close() # Do not forget to close! The backend server will consume a lot memery

