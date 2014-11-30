#!usr/bin/env python
#coding:utf-8
import numpy as np


def normalizeVec(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0

    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s

def normalizeMatrix(matrix):
    '''Normalize for each row in this matrix.'''
    row_num, column_num = np.shape(matrix)
    for row_index in range(row_num):
        normalizeVec(matrix[row_index])

class CAspectModel:
    def __init__(self):
        self.hidden_num = 0
        self.user_hidden_prob = None
        self.hidden_item_prob = None
        self.hidden_prob = None

    def initParameters(self, Data, hidden_num):
        '''
        Initialize the parameters of pLSA
        '''
        users_num, items_num = Data.shape
        self.hidden_num = hidden_num
        self.user_hidden_prob = np.random.random(size=(users_num, hidden_num)) #P(z|u)
        normalizeMatrix(self.user_hidden_prob)
        self.hidden_item_prob = np.random.random(size=(hidden_num, items_num)) #P(y|z)
        normalizeMatrix(self.hidden_item_prob)
        self.hidden_prob = np.zeros([users_num, items_num, hidden_num], dtype=np.float) #Q(z,u,y)

    def calParameters(self, Data, MAXIteration):
        '''
        Calculating the parameters.
        '''
        users_num, items_num = Data.shape
        for iteration in range(MAXIteration):
            print 'Iteration %s:' % iteration
            print '*'*100
            print self.user_hidden_prob
            print self.hidden_item_prob
            print '\tE Step:', iteration
            for user_id in range(users_num):
                for item_id in range(items_num):
                    prob = self.user_hidden_prob[user_id, :] * self.hidden_item_prob[:, item_id]

                    normalizeVec(prob)
                    self.hidden_prob[user_id][item_id] = prob

            print '\tM Stemp:', iteration
            #Update p(z|u)
            for user_key in range(users_num):
                for hidden_key in range(self.hidden_num):
                    sum = 0.0
                    for item_key in range(items_num):
                        sum += Data[user_key][item_key]*self.hidden_prob[user_key][item_key][hidden_key]
                    self.user_hidden_prob[user_key][hidden_key] = sum
                normalizeVec(self.user_hidden_prob[user_key])

            #Update p(y|z)
            for hidden_key in range(self.hidden_num):
                for item_key in range(items_num):
                    sum = 0.0
                    for user_key in range(users_num):
                        sum += Data[user_key][item_key]*self.hidden_prob[user_key][item_key][hidden_key]
                    self.hidden_item_prob[hidden_key][item_key] = sum
                normalizeVec(self.hidden_item_prob[hidden_key])

    def calSubRecommend(self, Data, active_id):
        '''
        Calculate the recommendation for the active_user_index
        '''
        recommendation = {}
        for item_id in range(len(Data[active_id])):
            if Data[active_id][item_id] != 0: continue
            recommendation[item_id] = self.user_hidden_prob[active_id,:]*self.hidden_item_prob[:,item_id]
        return recommendation

    def calRecommend(self, _Data, hidden_num, MAXIteration, top_num):
        '''
        Calculating the recommendation for each user.
        Data: size:M*N, M represents the number of users, N represents the number of items.
        Data represents the graph between users and items.
        '''
        Data = np.array(_Data)
        self.initParameters(Data, hidden_num)
        self.calParameters(Data, MAXIteration)
        recommendation = {}
        users_num, items_num = Data.shape
        for user_id in range(users_num):
            temp = self.calSubRecommend(Data, user_id)
            temp = sorted(temp.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[self.usersList[user_id]] = [item_id for item_id, prob in temp[:top_num]]
        return recommendation

if __name__=='__main__':
    from CommonModules import loadData
    data = loadData()
    mCUserBasedCF = CAspectModel()
    recommendation = mCUserBasedCF.calRecommend(data, 2, 100, 1)
