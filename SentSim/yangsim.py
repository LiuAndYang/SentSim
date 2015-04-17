# coding=utf8
# Copyright (c) 2014, Ryan Liu@HIT
# All rights reserved.
import sys
import numpy
import Levenshtein
import nltk
from collections import Counter, defaultdict
from sklearn import linear_model
from sklearn import svm
import nltk
from numpy.linalg import norm
from nltk.corpus  import stopwords
from nltk.corpus.reader import CorpusReader
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
import re
import math
import cPickle as pickle
import os

if len(sys.argv) != 8:
	print>>sys.stderr,"Usage: %s , sentences.txt,scores.txt and test.txt are needed!" %sys.argv[0]
	exit(1)
#Get the sentences into a list
sentcs = []
for l in open(sys.argv[1]):
    sentcs.append([x.strip() for x in l.lower().split("\t")])
#Get the scores into a list
SCORES = []
for s in open(sys.argv[2]):
    SCORES.append(float(s))
#Get the text into a list
teststcs1 = []
teststcs2 = []
teststcs3 = []
teststcs4 = []
teststcs5 = []
teststcs  = []
for l in open(sys.argv[3]):
    teststcs.append([x.strip() for x in l.split("\t")])
    teststcs1.append([x.strip() for x in l.lower().split("\t")])
for l in open(sys.argv[4]):
    teststcs.append([x.strip() for x in l.split("\t")])
    teststcs2.append([x.strip() for x in l.lower().split("\t")])
for l in open(sys.argv[5]):
    teststcs.append([x.strip() for x in l.split("\t")])
    teststcs3.append([x.strip() for x in l.lower().split("\t")])
for l in open(sys.argv[6]):
    teststcs.append([x.strip() for x in l.split("\t")])
    teststcs4.append([x.strip() for x in l.lower().split("\t")])
for l in open(sys.argv[7]):
    teststcs.append([x.strip() for x in l.split("\t")])
    teststcs5.append([x.strip() for x in l.lower().split("\t")])
trainlen = len(sentcs)
sentcs = sentcs + teststcs1+ teststcs2+ teststcs3+ teststcs4+ teststcs5
pickle_sentcs_path = '/home/yang/YangSim/pickles/Models/sentcs.pkl'
print("Dump the sentences into pickle.\n")
with open(pickle_sentcs_path, 'wb') as f:
    pickle.dump(teststcs, f)
print "sentcs Dumped!"
print len(sentcs)
print '\n'
print trainlen
####################Gloable_variable###############
to_wordnet_tag = {
        'NN':wordnet.NOUN,
        'JJ':wordnet.ADJ,
        'VB':wordnet.VERB,
        'RB':wordnet.ADV
    } #只选择名词，动词，形容词,可能是只有这几种存在源词吧。
wordnet_sim_function = ['wup_similarity','res_similarity','path_similarity','lin_similarity','lch_similarity','jcn_similarity']
tokzd_sentcs = [] #用来存储标注过词性的句子的列表
lemtazd_sentcs = []#只保留了词源形式的
#load the idf weight
IDF = dict()
Allwords = dict()
if os.path.isfile('/home/yang/YangSim/pickles/Datas/IDF.pkl'): 
    with open('/home/yang/YangSim/pickles/Datas/IDF.pkl', 'rb') as f:
        IDF = pickle.load(f)
#load the word frequency weight
wf_weigth = defaultdict(float)
###################TOOLS_PREPROCESSING##############
def get_lemmatized_words(sa):  #就是找到了某些词的词根而已

	rez = []
	for w, wpos in sa:
		w = w.lower()   #小写它。。。
		if w in stopwords.words("english") : #去除了停用词了
			continue
		wtag = to_wordnet_tag.get(wpos[:2]) #只取标注的前两个字母
		if wtag is None:
			wlem = w  #无法找到变形的，
		else:
			wlem = wordnet.morphy(w, wtag) or w #如果在前面找不到同源词，那么就返回原词
		rez.append(wlem)
	return rez
class Sim:
    def __init__(self, words, vectors):
        self.word_to_idx = {a: b for b, a in
                            enumerate(w.strip() for w in open(words))} #（词，序号）
        self.mat = numpy.loadtxt(vectors) #读取向量文本

    def bow_vec(self, b):                #这里的b就是一个lemmarized的词的字典，value都为1
        vec = numpy.zeros(self.mat.shape[1])
        for k, v in b.iteritems():
            idx = self.word_to_idx.get(k, -1) #是字典的get函数，没有找到的话就返回-1呀，孩纸。
            if idx >= 0:  #这个序号对应的行向量。
                vec += self.mat[idx] / (norm(self.mat[idx]) + 1e-8) * v #防止为0
        return vec          #转换成了潜在的语义向量空间里的向量

    def calc(self, b1, b2):   #cos相似度的经典计算方法呀，cos公式(n1*m1 + n2*m2 ...)/\n\/\m\
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        return abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8))
wiki_sim = Sim('voc', 'lsaModel')
#load the word frequency
def load_wweight_table(path):
    lines = open(path).readlines()
    wweight = defaultdict(float)
    if not len(lines):
        return (wweight, 0.)
    totfreq = int(lines[0])
    for l in lines[1:]:
        w, freq = l.split()
        freq = float(freq)
        if freq < 10:
            continue
        wweight[w] = math.log(totfreq / freq)

    return wweight
wf_weigth = load_wweight_table('word-frequencies.txt')
#load the word2vec,only keep the word that has appeared, because of the limit of the memory
W2Vdict = defaultdict(list)
def load_word2vec(filename): #'word2vec'
    f = open(filename)
    f.readline()
    print 'Reading the word2vec file'
    for line in f:
        templist = line.split()
        if Allwords.has_key(templist[0]):
            W2Vdict[templist[0]] = [float(x) for x in templist[1:]]
    print W2Vdict['man'][0:14]
################# GENERATE FEATURES################
#1.TEXT similarity
#编辑距离
t_edit_score = []
t_edit_score_norm = []
#print sentcs[6862]
#print sentcs[6861]
for index,[s1,s2] in enumerate(sentcs):
    #print index
    #print
    t_edit_score.append(Levenshtein.distance(s1,s2))
#编辑距离的归一化
tempmax = max(t_edit_score)
for score in t_edit_score:
	score = score*1.0/tempmax
	t_edit_score_norm.append(score)
#Jaccard distance
t_jaccard_score = []
for [s1,s2] in sentcs:
	Tagged_s1 = []
	Tagged_s2 = []
	s1_set    = []
	s2_set    = []
	Tagged_s1 = nltk.pos_tag(nltk.word_tokenize(s1))
	Tagged_s2 = nltk.pos_tag(nltk.word_tokenize(s2))
	tokzd_sentcs.append([Tagged_s1,Tagged_s2]) # 对全局的标注变量赋值
	for (w,t) in Tagged_s1:
		s1_set.append(t)
	for (w,t) in Tagged_s2:
		s2_set.append(t)
	t_jaccard_score.append(len(set(s1_set) & set(s2_set))*1.0/len(set(s1_set) | set(s2_set)))
#2.lexical substitution USING wordnet
t_wn_score = []
t_wnres_score = []
t_wnres_score_norm = []
t_wnlch_score = []
t_wnlch_score_norm = []
t_wnlin_score = []
t_wnwup_score = []
t_wnjcn_score = []
t_wnjch_score_norm = []
wpathsimcache = {} #作为缓存使用，如果已经计算过了就不用再算了，节省时间，重复的情况很多么？
brown_ic = wordnet_ic.ic('ic-brown.dat')
shaks_ic = wordnet_ic.ic('ic-shaks.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
rpathsimcache = {}
lpathsimcache = {}
upathsimcache = {}
jpathsimcache = {}
npathsimcache = {}
def wpathsim(a, b, c):  #计算两个单词的路径语义相似度，，，吗？
    if a > b:
        b, a = a, b
    p = (a, b)
    if a == b:
        wpathsimcache[p] = 1.
        rpathsimcache[p] = 1.
        lpathsimcache[p] = 1.
        return 1.
    sa = wordnet.synsets(a) #同义词集合，放入sa中
    sb = wordnet.synsets(b)
    if c == 0:
    	if p in wpathsimcache:
    		return wpathsimcache[p]
        mx = max([wa.path_similarity(wb)
              for wa in sa
              for wb in sb
              ] + [0.])   #在列表末尾合并一个0.00以防都是none-type
        wpathsimcache[p] = mx
    if c == 1:
    	if p in rpathsimcache:
    		return rpathsimcache[p]
    	tempsim = []
    	for wa in sa:
    		for wb in sb:
    			if (wa.pos == wb.pos and wa.pos in ['n','v']):
    				tempsim.append(wa.res_similarity(wb,brown_ic))
        mx = max(tempsim + [0.])   #在列表末尾合并一个0.00以防都是none-type
        if mx > 1000:
        	mx = 1.0
        rpathsimcache[p] = mx
    if c == 2:
    	if p in lpathsimcache:
    		return lpathsimcache[p]
    	tempsim = []
    	for wa in sa:
    		for wb in sb:
    			if (wa.pos == wb.pos and wa.pos in ['n','v']):
    				tempsim.append(wa.lch_similarity(wb))
        mx = max(tempsim + [0.])   #在列表末尾合并一个0.00以防都是none-type
        if mx > 1000:
        	mx = 1.0
        lpathsimcache[p] = mx
    if c == 3:
    	if p in jpathsimcache:
    		return jpathsimcache[p]
    	tempsim = []
    	for wa in sa:
    		for wb in sb:
    			if (wa.pos == wb.pos and wa.pos in ['n','v']):
    				tempsim.append(wa.jcn_similarity(wb,brown_ic))
        mx = max(tempsim + [0.])   #在列表末尾合并一个0.00以防都是none-type
        if mx > 1000:
        	mx = 1.0
        jpathsimcache[p] = mx
    if c == 4:
    	if p in npathsimcache:
    		return npathsimcache[p]
    	tempsim = []
    	for wa in sa:
    		for wb in sb:
    			if (wa.pos == wb.pos and wa.pos in ['n','v']):
    				tempsim.append(wa.lin_similarity(wb,brown_ic))
        mx = max(tempsim + [0.])   #在列表末尾合并一个0.00以防都是none-type
        if mx > 1:
        	print 'lin out of range'
        npathsimcache[p] = mx
    if c == 5:
    	if p in upathsimcache:
    		return upathsimcache[p]
        mx = max([wa.wup_similarity(wb)
              for wa in sa
              for wb in sb
              ] + [0.])   #在列表末尾合并一个0.00以防都是none-type
        upathsimcache[p] = mx
    return mx
def calc_wn_prec(lema, lemb, sim_measure):
    rez = 0.
    for a in lema:
        ms = 0.
        for b in lemb:
            #print(wpathsim(a, b))
            ms = max(ms,wpathsim(a, b, sim_measure))
        rez += ms
    return rez / len(lema)

def wn_sim_match(lema, lemb, sim_measure): #wordnet 相似度匹配函数
    f1 = 1. #就是一个float类型的数
    p = 0.
    r = 0.
    if len(lema) > 0 and len(lemb) > 0:
        p = calc_wn_prec(lema, lemb, sim_measure)
        r = calc_wn_prec(lemb, lema, sim_measure)
        f1 = 2. * p * r / (p + r) if p + r > 0 else 0.  #此处为什么要用f1 score？原来是为了调和呀，哈哈 。
    return f1
#############
for [la,lb] in tokzd_sentcs:
	la = get_lemmatized_words(la)
	lb = get_lemmatized_words(lb)
	lemtazd_sentcs.append([la,lb])
	t_wn_score.append(wn_sim_match(la,lb, 0))
	t_wnres_score.append(wn_sim_match(la,lb, 1))
	t_wnlch_score.append(wn_sim_match(la,lb, 2))
	t_wnjcn_score.append(wn_sim_match(la,lb, 3))
	t_wnlin_score.append(wn_sim_match(la,lb, 4))
	t_wnwup_score.append(wn_sim_match(la,lb, 5))
#norm the t_wnres_scores
tempmax = max(t_wnres_score)
for score in t_wnres_score:
	score = score*1.0/tempmax
	t_wnres_score_norm.append(score)
#norm the t_wnlch_score
tempmax = max(t_wnlch_score)
for score in t_wnlch_score:
	score = score*1.0/tempmax
	t_wnlch_score_norm.append(score)
#norm the t_wnjcn_score
tempmax = max(t_wnjcn_score)
for score in t_wnjcn_score:
	score = score*1.0/tempmax
	t_wnjch_score_norm.append(score)
t_lsa_score = []
#3.LSA similarity 使用lemtazd_sentcs里保存了源格式的
for [la,lb] in lemtazd_sentcs:
	#lsavec_la = fetchlasvec(la)
	#lsavec_lb = fetchlasvec(lb)
    wa = Counter(la) #变成了count类型的词典，就想一个词向量一样。
    wb = Counter(lb)
    d1 = {x:1 for x in wa} #只是找到这个词，并没有利用它的计数，句子太短，加上也没什么用，是一个字典。。。
    d2 = {x:1 for x in wb}
    dictMerged=dict(d1, **d2)
    Allwords = dict(Allwords, **dictMerged)
    t_lsa_score.append(wiki_sim.calc(d1, d2)) #是使用了LSA的方法呀。
t_w2v_score = []
unfind = []
cntvalid = 0
cntallwords = 0
cntallwords2 = 0
cntvalid2 = 0
#4.The word2vector similarity
load_word2vec('word2vec')
for [la,lb] in lemtazd_sentcs:
    wa = Counter(la) #变成了count类型的词典，就想一个词向量一样。
    wb = Counter(lb)
    #d1 = [x for x in wa] #只是找到这个词，并没有利用它的计数，句子太短，加上也没什么用，是一个字典。。。
    #d2 = [x for x in wb]
    v1 = numpy.zeros(200)
    v2 = numpy.zeros(200)
    for x in wa:
    	cntallwords = cntallwords + 1
        d1 = W2Vdict.get(x,-1)
        if d1!= -1:
            v1 = v1 + numpy.array(d1)
            cntvalid = cntvalid + 1
        else:
        	unfind.append(x)
    for x in wb:
        d2 = W2Vdict.get(x,-1)
        cntallwords2 = cntallwords2+1
        if d2!= -1:
            cntvalid2 = cntvalid2 +1
            v2 = v2 + numpy.array(d2)
        else:
        	unfind.append(x)
    t_w2v_score.append(abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8)))
print cntallwords,cntvalid
print cntallwords2,cntvalid2
#get the weighted lsa similarity
t_w_lsa_score = []
t_wf_lsa_score = []
for [la,lb] in lemtazd_sentcs:
    wa = Counter(la) #变成了count类型的词典，就想一个词向量一样。
    wb = Counter(lb)
    d1 = {x:IDF.get(x,0) for x in wa} #只是找到这个词，并没有利用它的计数，句子太短，加上也没什么用，是一个字典。。。
    d2 = {x:IDF.get(x,0) for x in wb}
    d3 = {x:wf_weigth.get(x,0) for x in wa} #只是找到这个词，并没有利用它的计数，句子太短，加上也没什么用，是一个字典。。。
    d4 = {x:wf_weigth.get(x,0) for x in wb}
    t_w_lsa_score.append(wiki_sim.calc(d1, d2)) #是使用了LSA的方法呀。
    t_wf_lsa_score.append(wiki_sim.calc(d3, d4)) #是使用了LSA的方法呀。
#Combine the features
FEATURES = [[a,b] for a,b in zip(t_edit_score_norm,t_jaccard_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wn_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wnres_score_norm)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wnlch_score_norm)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wnjch_score_norm)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wnlin_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wnwup_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_lsa_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_w2v_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_w_lsa_score)]
FEATURES = [a+[b] for a,b in zip(FEATURES,t_wf_lsa_score)]
print(FEATURES[147])
#print(type(FEATURES[147][3]))
fea = open('features.txt','w')
for scr in FEATURES:
	print>>fea,scr
fea.close
print("Start to train the model.\n")
clf = svm.SVR(C=1, cache_size=200, coef0=0.0, degree=3,
epsilon=0.1, gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
random_state=None, shrinking=True, tol=0.001, verbose=False)
clf.fit(FEATURES[0:trainlen],SCORES)
pickle_model_path = '/home/yang/YangSim/pickles/Models/clf.pkl'
pickle_fea_path = '/home/yang/YangSim/pickles/Features/tesfea.pkl'
print("Dump the model into pickle.\n")
with open(pickle_model_path, 'wb') as f:
    pickle.dump(clf, f)
print "CLF Dumped!"
print("Dump the features into pickle.\n")
with open(pickle_fea_path, 'wb') as f:
    pickle.dump(FEATURES[trainlen:], f)
print "FEATURES Dumped!"
#lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#lr.fit(FEATURES[0:trainlen],SCORES)
#lr.fit(FEATURES[0:trainlen],SCORES)
print("Predict the test file MSRpar.\n")
MSRpar = trainlen + len(teststcs1)
RESULTS = clf.predict(FEATURES[trainlen:MSRpar])
#RESULTS = lr.predict(FEATURES[trainlen:])
#Write the result into txt
f = open('predict_MSRpar.txt','w')
for scr in RESULTS:
	print>>f,scr
f.close
print("Predict the test file MSRvid.\n")
MSRvid = MSRpar + len(teststcs2)
RESULTS = clf.predict(FEATURES[MSRpar:MSRvid])
#RESULTS = lr.predict(FEATURES[trainlen:])
#Write the result into txt
f = open('predict_MSRvid.txt','w')
for scr in RESULTS:
    print>>f,scr
f.close
print("Predict the test file SMTeuroparl.\n")
SMTeuroparl = MSRvid + len(teststcs3)
RESULTS = clf.predict(FEATURES[MSRvid:SMTeuroparl])
#RESULTS = lr.predict(FEATURES[trainlen:])
#Write the result into txt
f = open('predict_SMTeuroparl.txt','w')
for scr in RESULTS:
    print>>f,scr
f.close
print("Predict the test file OnWn.\n")
OnWn = SMTeuroparl + len(teststcs4)
RESULTS = clf.predict(FEATURES[SMTeuroparl:OnWn])
#RESULTS = lr.predict(FEATURES[trainlen:])
#Write the result into txt
f = open('predict_OnWn.txt','w')
for scr in RESULTS:
    print>>f,scr
f.close
print("Predict the test file SMTnews.\n")
SMTnews = OnWn + len(teststcs5)
RESULTS = clf.predict(FEATURES[OnWn:SMTnews])
#RESULTS = lr.predict(FEATURES[trainlen:])
#Write the result into txt
f = open('predict_SMTnews.txt','w')
for scr in RESULTS:
    print>>f,scr
f.close
print 'The unfind words are'
f = open('unfind.txt','w')
for scr in unfind:
    print>>f,scr
f.close

