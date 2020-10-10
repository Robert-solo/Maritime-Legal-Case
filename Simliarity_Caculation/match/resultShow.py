# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class ResultShow(object):
    def __init__(self):
        self.lstep = 0
        self.plv = []
        self.rlv = []
        self.rlen = 0
        self.xpassage = []
        
        
    def load_Result(self,plv,rlv,sni1_plv,sni1_rlv,sni2_plv,sni2_rlv,lstep =1 ):
        self.lstep = lstep
        self.plv = plv
        self.rlv = rlv
        self.sni1_plv = sni1_plv
        self.sni1_rlv = sni1_rlv
        self.sni2_plv = sni2_plv
        self.sni2_rlv = sni2_rlv
        self.rlen = len(plv)
        self.xpassage = [ (i+1) * lstep for i in range(self.rlen)]
        
        
    def show_mAP(self,plv,rlv,sni1_plv,sni1_rlv,sni2_plv,sni2_rlv,data,ex):
        '''
        x=[i for i in range(1,1000,1)]
        y1=[(t/100)**2 for t in x]
        y2=[(t/100)**1.5 + 1 for t in x]
        '''
        plt.figure(figsize=(16,8)) 
        #plt.xlim(xmax=2000, xmin=0)
        #plt.ylim(ymax=120, ymin=0)
        #plt.title("mAP comperation in Med",fontsize = 20)
        plt.title(" ",fontsize = 20)
        plt.ylabel("Precision", fontsize = 20)
        plt.xlabel("Recall", fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        
        plt.plot(rlv, plv, 'r-', marker = 'o')
        #plt.plot(sni1_rlv, sni1_plv, 'b-')
        plt.plot(sni1_rlv, sni1_plv, 'g-',marker = 'v')
        plt.legend(['BM25', 'TSM_BM25'],loc = 'upper right',fontsize = 20)
        plt.savefig('../support/mAP_'+ data +'_' + ex + '.jpg', dpi = 600)
        plt.show()
        
        
        
        
        
        
        
        
    def show_F1(self,q,data,Match_fun,ex):
        plt.figure(figsize=(12,6),dpi=80) 
        
        
        
        plt.xlabel('x:sum of query')
        plt.ylabel('y:F1 score')
        #plt.title('Titles: '+ "Med" +' '+ Match_fun +' F1')
        plt.title(' ')
        m = Match_fun
        q -= 1
        
        q_10pre = [ self.plv[i][q] for i in range(self.rlen)]
        temp = q_10pre
        q_10pre = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        q_10rec = [ self.rlv[i][q] for i in range(self.rlen)]
        temp = q_10rec
        q_10rec = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        sni_q_10pre1 = [self.sni1_plv[i][q] for i in range(self.rlen)]
        temp = sni_q_10pre1
        sni_q_10pre1 = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        sni_q_10rec1 = [self.sni1_rlv[i][q] for i in range(self.rlen)]
        temp = sni_q_10rec1
        sni_q_10rec1 = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        sni_q_10pre2 = [self.sni2_plv[i][q] for i in range(self.rlen)]
        temp = sni_q_10pre2
        sni_q_10pre2 = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        sni_q_10rec2 = [self.sni2_rlv[i][q] for i in range(self.rlen)]
        temp = sni_q_10rec2
        sni_q_10rec2 = [ sum(temp[0:i+1])/(i+1) for i in range(self.rlen)]
        
        print(sni_q_10rec1)
        
    
        compute_F1 = []
        compute_F1_Sni1 = []
        compute_F1_Sni2 = []
        for i in range(0,self.rlen,1):
            compute_F1.append(2*q_10pre[i]*q_10rec[i]/(q_10pre[i]+q_10rec[i]+0.00001))
            compute_F1_Sni1.append(2*sni_q_10pre1[i]*sni_q_10rec1[i]/(sni_q_10pre1[i]+sni_q_10rec1[i]+0.00001))
            compute_F1_Sni2.append(2*sni_q_10pre2[i]*sni_q_10rec2[i]/(sni_q_10pre2[i]+sni_q_10rec2[i]+0.00001))
        #print("F1:\n", compute_F1,compute_F1_Sni1,compute_F1_Sni2)
        plt.xlim(0,self.rlen + self.lstep)  #  设置x轴刻度范围
        plt.ylim(max(min(compute_F1) - 0.2,0.0), min(max(compute_F1)+0.3,1.0))   # 设置y轴刻度的范围
        #plt.xticks([x for x in range(max(self.xpassage)+self.lstep * 2) if x % self.lstep == 0])  # x标记的step
        plt.xticks(range(0,self.rlen + self.lstep,5))
        
        plt.plot(self.xpassage,compute_F1,color='r',linestyle='-',linewidth = '2',marker = 'o', markerfacecolor='black',markersize = 3)  
        #plt.plot(self.xpassage,compute_F1_Sni1,color='g',linestyle='-.',linewidth = '2',marker = 'o', markerfacecolor='black',markersize = 3)  
        plt.plot(self.xpassage,compute_F1_Sni1,color='b',linestyle='-',linewidth = '2',marker = 'o', markerfacecolor='black',markersize = 3) 
        #plt.legend([m, 'TSM_'+m +'(v1)', 'TSM_'+m +'(v2)'],loc = 'upper left')
        plt.legend([m, 'TSM_'+m],loc = 'upper left')
        '''
        plt.legend(
                ['VNBM25','Sni_BM25(max+ave)','Sni_BM25(fix_wei)',
                 'WMD','WMD(max+ave)','WMD(fix_wei)',
                 'lmj','lmj(max+ave)','lmj(fix_wei)',
                 'jac','jac(max+ave)','jac(fix_wei)'],
                 loc = 'upper left')
        '''
        
        plt.savefig('../support/F1_'+ data +'_'+ Match_fun + ' ' + ex + '.jpg')
        plt.show()  #绘制图像
    
    def show_PR(self,):
        plt.figure(figsize=(12,6),dpi=80) 

        plt.xlim(0,1)  #  设置x轴刻度范围，从0~1000
        
        plt.ylim(0,1)   # 设置y轴刻度的范围，从0~20
        plt.xlabel('x:Recovery')
        plt.ylabel('y:Precision')
        
        #med
        r = [0.177, 0.313, 0.5, 0.601]  
        p = [0.724, 0.638, 0.531, 0.432]
 
        r1 = [0.194,0.317, 0.477, 0.568] 
        p1 = [0.76,  0.643, 0.503, 0.409]
        r2 = [0.183,0.315,0.478,0.561] 
        p2 = [0.727,0.64, 0.5,0.403]
        
        
        print(self.rlv, self.plv, self.sni1_rlv, self.sni1_plv, self.sni2_rlv, self.sni2_plv)
        plt.plot(r, p, 'g-')
        plt.plot(r1, p1, 'b-')
        plt.plot(r2, p2, 'r-')
        plt.title('Titles:nlp_q10_F1')

        
        pass
    
    
    #测试用
    def show_dispa(self,pl,rl,sni1_pl,sni1_rl,q=0):
        
        qlen = len(pl)
        slen = len(pl[0])
        #print(pl)
        #print(sni1_pl)
        pdis_list = []
        top20_list = []
        for index in range(qlen):
            dis = 0
            for i in range(slen):
                dis += (pl[index][i]-sni1_pl[index][i])
            dis = round(dis,2)
            top20_list.append((index,pl[index][0],sni1_pl[index][0],pl[index][1],sni1_pl[index][1],pl[index][2],sni1_pl[index][2]))
            pdis_list.append((index, dis))
        pdis_list.sort(key=lambda x:x[1], reverse=True)
        
        return pdis_list,top20_list
        
            
            
        