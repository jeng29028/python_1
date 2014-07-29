# 原作者 Neil Schemenauer <nas@arctrix.com>

import math
import random
import string

random.seed() #不固定每次亂數取法


def 建矩陣(I,J,fill=0.0):
        m = []
        for i in range(I):
            m.append([fill]*J)
        return m


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y**2

#..........................................................
class 神經網路:
    def __init__(self, 輸入節點數, 隱藏層節點數, 輸出節點數):
        # 三層之節點數目
        self.輸入節點數 = 輸入節點數
        self.隱藏層節點數 = 隱藏層節點數
        self.輸出節點數 = 輸出節點數

        # 節點激發的結果
        self.輸入點激發結果 = [1.0]*self.輸入節點數
        self.隱藏點激發結果 = [1.0]*self.隱藏層節點數
        self.輸出點激發結果 = [1.0]*self.輸出節點數

        # 設置權重
        self.輸入節點權重 = 建矩陣(self.輸入節點數, self.隱藏層節點數)
        self.輸出節點權重 = 建矩陣(self.隱藏層節點數, self.輸出節點數)


        # 亂數初始權重
        for i in range(self.輸入節點數):
            for j in range(self.隱藏層節點數):
                self.輸入節點權重[i][j] = random.random()

        for j in range(self.隱藏層節點數):
            for k in range(self.輸出節點數):
                self.輸出節點權重[j][k] = random.random()

#............
    def 更新(self, 輸入值):

        # 激發輸入層
        for i in range(self.輸入節點數):
            self.輸入點激發結果[i] = 輸入值[i]

        # 激發隱藏層
        for j in range(self.隱藏層節點數):
            加總結果 = 0.0
            for i in range(self.輸入節點數):
                加總結果 = 加總結果 + self.輸入點激發結果[i] * self.輸入節點權重[i][j]
            self.隱藏點激發結果[j] = sigmoid(加總結果)

        # 激發輸出層
        for k in range(self.輸出節點數):
            加總結果 = 0.0
            for j in range(self.隱藏層節點數):
                加總結果 = 加總結果 + self.隱藏點激發結果[j] * self.輸出節點權重[j][k]
            self.輸出點激發結果[k] = sigmoid(加總結果)

        return self.輸出點激發結果[:]
#.....................

    def 倒傳遞(self, 目標值, 學習率):

        # 計算輸出層誤差
        輸出層變化 = [0.0] * self.輸出節點數
        for k in range(self.輸出節點數):
            誤差 = 目標值[k]-self.輸出點激發結果[k]
            輸出層變化[k] = dsigmoid(self.輸出點激發結果[k]) * 誤差

        # 計算隱藏層誤差
        隱藏層變化 = [0.0] * self.隱藏層節點數
        for j in range(self.隱藏層節點數):
            誤差 = 0.0
            for k in range(self.輸出節點數):
                誤差 = 誤差 + 輸出層變化[k]*self.輸出節點權重[j][k]
            隱藏層變化[j] = dsigmoid(self.隱藏點激發結果[j]) * 誤差

        # 更新輸出權重
        for j in range(self.隱藏層節點數):
            for k in range(self.輸出節點數):
                變量 = 輸出層變化[k]*self.隱藏點激發結果[j]
                self.輸出節點權重[j][k] = self.輸出節點權重[j][k] + 學習率*變量*self.隱藏點激發結果[j]

        # 更新輸入權重
        for i in range(self.輸入節點數):
            for j in range(self.隱藏層節點數):
                變量 = 隱藏層變化[j]*self.輸入點激發結果[i]
                self.輸入節點權重[i][j] = self.輸入節點權重[i][j] + 學習率*變量*self.輸入點激發結果[i]

        # 計算誤差
        誤差 = 0.0
        for k in range(len(目標值)):
            誤差 = 誤差 + 0.5*(目標值[k]-self.輸出點激發結果[k])**2
        return 誤差


    def 測驗(self, patterns):
        for p in patterns:
            print(p[0], '->', self.更新(p[0]))


    def 訓練(self, patterns, 訓練次數=10000, 學習率=0.5):
        for i in range(訓練次數):
            誤差 = 0.0
            for p in patterns:
                輸入值 = p[0]
                目標值 = p[1]
                self.更新(輸入值)
                誤差 = 誤差 + self.倒傳遞(目標值, 學習率)
            if i % 100 == 0:            #每訓練100次輸出誤差一次
                print('誤差 %-.3f' % 誤差)
#..................................................................

def demo():
    #XOR 樣本
    樣本 = [
             [[0,0], [0]],
             [[0,1], [1]],
             [[1,0], [1]],
             [[1,1], [0]],
           ]

    # 建置2個輸入 2個隱藏 1個輸出 之神經網路
    n = 神經網路(2, 2, 1)

    n.訓練(樣本)

    n.測驗(樣本)


if __name__ == '__main__':
    demo()



