#!usr/bin/env python
#coding:utf-8
import numpy as np

class CUserBasedCF:
    def __init__(self):
        pass

    def calSubRecommend(self, data, active_user_id):
        avr_UserA = float(np.sum(data[active_user_id]))/len(data[active_user_id])
        recommendation = np.ones(len(data[active_user_id]), 'float') * avr_UserA
        for user_id in range(len(data)):
            if user_id == active_user_id:
                continue
            recommendation += self.calSimilarity(data[active_user_id], data[user_id])

        temp = {}
        for item_id in range(len(recommendation)):
            if data[active_user_id][item_id] == 0:
                temp[item_id] = recommendation[item_id]
        return temp

    def calSimilarity(self, userA, userB):
        avr_UserA = float(np.sum(userA))/len(userA)
        avr_UserB = float(np.sum(userB))/len(userB)
        temp_UserA = userA - avr_UserA
        temp_UserB = userB - avr_UserB
        sim = temp_UserA*temp_UserB/np.linalg.norm(temp_UserA)/np.linalg.norm(temp_UserB)
        return sim*temp_UserB

    def calRecommend(self, mData, top_num):
        recommendation = {}
        data = np.array(mData)
        for user_id in range(len(data)):
            temp = self.calSubRecommend(data, user_id)
            temp = sorted(temp.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[user_id] = [item_id for item_id, rate in temp[:top_num]]
        return recommendation


if __name__=='__main__':
    from CommonModules import loadData
    data = loadData()
    mCUserBasedCF = CUserBasedCF()
    recommendation = mCUserBasedCF.calRecommend(data, 1)
    print