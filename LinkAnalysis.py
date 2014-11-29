#!usr/bin/env python
#coding:utf-8

import numpy as np

def normalize(matrix):
    '''Normalizing the matrix.'''
    matrix_sum = matrix.sum(1)
    for i in range(len(matrix_sum)):
        matrix[i] = 1.0*matrix[i]/matrix_sum[i]

class CLinkAnalysis:
    def __init__(self):
        ''''''
        self.gamma = 0.9
        self.eta = 1.0
        self.A = None
        self.B = None
        self.CR0 = None
        self.PR = None
        self.CR = None

    def calPR(self):
        '''
        Calculating PR = A'*CR
        '''
        self.PR = np.dot(self.A.transpose(), self.CR)

    def calCR(self):
        '''
        Calculating CR = B(A)*PR + CR0
        '''
        self.CR = np.dot(self.B, self.PR)
        normalize(self.CR)
        self.CR = self.CR + self.CR0

    def initParameters(self, Data):
        '''
        Initializing the parameters, including A, B, CR0
        '''
        self.A = np.array(Data)
        self.B = np.zeros(self.A.shape, 'float')
        for i in range(self.A.shape[0]):
            temp = np.sum(self.A[i,:])
            for j in range(self.A.shape[1]):
                self.B[i,j] = float(self.A[i,j])/(temp**self.gamma)
        self.CR0 = np.eye(N=self.A.shape[0], M=self.A.shape[1])*self.eta
        self.CR = self.CR0


    def calParameters(self, MAXIteration):
        '''
        Calculating the parameters.
        '''
        for t in range(MAXIteration):
            self.PR = self.calPR()
            self.CR = self.calCR()

    def calSubRecommend(self, Data, active_id):
        '''
        Calculating the recommendation for the active_id.
        '''
        recommendation = {}
        for i in range(self.PR.shape[1]):
            if Data[active_id][i] != 0: continue
            recommendation[i] = self.PR[active_id][i]
        return recommendation

    def calRecommend(self, Data, MAXIteration, top_num):
        '''
        Calculating the recommendation for each user.
        Data: size:M*N, M represents the number of users, N represents the number of items.
        Data represents the graph between users and items.
        '''
        self.initParameters(Data)
        self.calParameters(MAXIteration)
        recommendation = {}
        for i in range(Data.shape[0]):
            temp = self.calSubRecommend(Data, i)
            temp = sorted(temp.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[i] = [item_id for item_id, sim in temp[:top_num]]
        return recommendation



if __name__=='__main__':
    from CommonModules import loadData
    data = loadData()
    mCUserBasedCF = CLinkAnalysis()
    recommendation = mCUserBasedCF.calRecommend(data, 3, 1)
    print recommendation
    print '*'*100
    print mCUserBasedCF.PR
    print '*'*100
    print mCUserBasedCF.CR