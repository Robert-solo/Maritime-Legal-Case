#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:42:12 2019

@author: leyv
"""


"""
import matplotlib.pyplot as plt
import numpy as np

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)

# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)

# 柱子总数
N = 7
# 包含每个柱子对应值的序列
values = (25, 32, 34, 20, 41, 50, 65)

# 包含每个柱子下标的序列
index = np.arange(N)

# 柱子的宽度
width = 0.35

# 绘制柱状图, 每根柱子的颜色为紫罗兰色
#2E8B57 深绿色
#1E90FF 浅蓝色
p2 = plt.bar(index, values, width, label="med", color="#87CEFA")

             
             
# 设置横轴标签
plt.xlabel('uplimit_ratio = 0.55')
# 设置纵轴标签
plt.ylabel('PR percentage')

# 添加标题
#plt.title('sensitivity analysis of uplimit rs_score in dataset Med')

plt.title('sensitivity analysis of filter score in dataset Med')




# 添加纵横轴的刻度
#plt.xticks(index, ('Jan', 'Fub', 'Mar', 'Apr', 'May', 'Jun'))
plt.xticks(index, ('2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'))
plt.yticks(np.arange(0, 81, 10))



# 添加图例
plt.legend(loc="upper right")

plt.show()
"""





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple



"""
fig, ax = plt.subplots()

'''
#Med
n_groups = 7
index = np.arange(n_groups)
bar_width = 0.35

Precision = [0.707, 0.72, 0.72, 0.747, 0.74, 0.733, 0.74]
Precision = [ i-0.5  for i in Precision]

Precision2 = [0.617, 0.617, 0.63, 0.633, 0.627, 0.627, 0.59]
Precision2 = [ i-0.5  for i in Precision2]

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'))
ax.set_xlabel('filter score')
ax.set_ylabel('Precision per')
ax.set_title(' ')


'''
#LISA
n_groups = 7
index = np.arange(n_groups)
bar_width = 0.35

Precision = [0.459, 0.471, 0.471, 0.482, 0.471, 0.459, 0.447]#, 0.424]
Precision = [ i-0.2  for i in Precision]

Precision2 = [0.329, 0.347, 0.341, 0.359, 0.365, 0.353, 0.359]#, 0.318]
Precision2 = [ i-0.2  for i in Precision2]

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'))
ax.set_xlabel('filter score')
ax.set_ylabel('Precision per')
ax.set_title(' ')




opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, Precision, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Top5')

rects2 = ax.bar(index + bar_width, Precision2, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Top10')



ax.set_yticks(np.arange(0, 0.40, step = 0.05) )
ax.set_yticklabels((0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55))
ax.legend()
fig.tight_layout()

import datetime
today = datetime.date.today()
#print(today)
plt.savefig('../support/save_fig/senti_ana-filter_score-'+str(today) +'LISA-P.jpg')
plt.show()



"""






fig, ax = plt.subplots(figsize=(12,8))


#Med
n_groups = 7
index = np.arange(n_groups)
bar_width = 1.0


Precision = [0.289, 0.293, 0.294, 0.305, 0.301, 0.297, 0.305]
Precision = [ i-0.2  for i in Precision]

#Precision2 = [0.409, 0.410, 0.418, 0.422, 0.415, 0.415, 0.395]
#Precision2 = [ i-0.3  for i in Precision2]


#Precision = [0.302, 0.297, 0.301, 0.3, 0.299, 0.299, 0.298]
#Precision = [ i-0.2  for i in Precision]

Precision2 = [0.402, 0.411, 0.41, 0.425, 0.407, 0.411, 0.409]
Precision2 = [ i-0.3  for i in Precision2]


Precision3 = [0.495, 0.508, 0.512, 0.515, 0.483, 0.485, 0.484]
Precision3 = [ i-0.4  for i in Precision3]


Precision4 = [0.489, 0.491, 0.497, 0.492, 0.466, 0.446, 0.417]
Precision4 = [ i-0.4 for i in Precision4]



ax.set_xticks(index + bar_width*3/10 )
ax.set_xticklabels(('2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'))
ax.set_xlabel('filter score')
ax.set_ylabel('F1 score')
ax.set_title(' ')


plt.ylabel('F1 score',fontsize = 20)
plt.xlabel('filter score',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#Med end



ax.set_xticks(index + bar_width*3/10 )
ax.set_xticklabels(( '3.8', '4.8', '5.8', '6.8', '7.8', '8.8','9.8'))
ax.set_xlabel('threshold control of related snippet')
ax.set_ylabel('F1 score')
ax.set_title(' ')


plt.ylabel('F1 score',fontsize = 20)
plt.xlabel('threshold for selecting related snippets',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)


opacity = 0.8
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, Precision, bar_width/5,
                alpha=opacity, color='c',
                error_kw=error_config,
                label='Top5')

rects2 = ax.bar(index + bar_width/5, Precision2, bar_width/5,
                alpha=opacity, color='y',
                error_kw=error_config,
                label='Top10')
rects3 = ax.bar(index + bar_width*2/5, Precision3, bar_width/5,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Top20')

rects4 = ax.bar(index + bar_width*3/5, Precision4, bar_width/5,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Top30')

ax.set_yticks(np.arange(0, 0.20, step = 0.05) )
ax.set_yticklabels((0.4,0.45,0.5,0.55))
ax.legend(fontsize = 18)
fig.tight_layout()

import datetime
today = datetime.date.today()
#print(today)
plt.savefig('../support/save_fig/senti_ana-uplimit_ratio-'+str(today) +'-MedF1.jpg',dpi = 600)
plt.show()


#LISA
fig, ax = plt.subplots(figsize=(12,8))

n_groups = 7
index = np.arange(n_groups)
bar_width = 1.0




#LISA end
Precision = [0.26,  0.281, 0.292, 0.296, 0.29,  0.289, 0.294]
Precision = [ i-0.2  for i in Precision]


Precision2 = [0.299, 0.311, 0.307, 0.329, 0.303, 0.303, 0.3]
Precision2 = [ i-0.2  for i in Precision2]


Precision3 = [0.298, 0.287, 0.282, 0.311, 0.298, 0.29,  0.283]
Precision3 = [ i-0.2  for i in Precision3]


Precision4 = [0.274, 0.265, 0.248, 0.266, 0.275, 0.265, 0.253]
Precision4 = [ i-0.2 for i in Precision4]



ax.set_xticks(index + bar_width*3/10 )
ax.set_xticklabels(('5.3', '6.3', '7.3', '8.3', '9.3', '10.3', '11.3'))
ax.set_xlabel('threshold control of related snippet')
ax.set_ylabel('F1 score')
ax.set_title(' ')


plt.ylabel('F1 score',fontsize = 20)
plt.xlabel('threshold for selecting related snippets',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)


opacity = 0.8
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, Precision, bar_width/5,
                alpha=opacity, color='c',
                error_kw=error_config,
                label='Top5')

rects2 = ax.bar(index + bar_width/5, Precision2, bar_width/5,
                alpha=opacity, color='y',
                error_kw=error_config,
                label='Top10')
rects3 = ax.bar(index + bar_width*2/5, Precision3, bar_width/5,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Top20')

rects4 = ax.bar(index + bar_width*3/5, Precision4, bar_width/5,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='Top30')

ax.set_yticks(np.arange(0, 0.20, step = 0.05) )
#ax.set_yticklabels((0.4,0.45,0.5,0.55))
ax.set_yticklabels((0.2,0.25,0.3,0.35))
ax.legend(fontsize = 18)
fig.tight_layout()

import datetime
today = datetime.date.today()
#print(today)
plt.savefig('../support/save_fig/senti_ana-uplimit_ratio-'+str(today) +'-LISAF1.jpg',dpi = 600)
plt.show()




