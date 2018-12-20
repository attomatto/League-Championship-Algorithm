#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LCA.py
Author: M. Ohmori
"""
""" 
<bibliography>
A. Husseinzadeh Kashan,
League Championship Algorithm (LCA): an algorithm for global optimization inspired by sport championships,
Eng. Appl. Soft Comput., 16 (2014), pp. 171-200.
"""

import numpy as np
import random
import math
from matplotlib import pyplot as plt
import datetime

L = 100  #league size 90
L_half = L / 2  #計算速度向上の為
S = 100  #season 1000
n = 10  #次元数
p_c = 0.3
PSI1 = 0.2
PSI2 = 1.0
schedule = list()


class LeagueChampionshipAlgorithm(object):
    def __init__(self):
        return

    def league(self, sentaku):
        if sentaku == 0:
            path_w = 'LCA_SphereResult.txt'
        elif sentaku == 1:
            path_w = 'LCA_RosenbrockResult.txt'
        wfile = open(path_w, mode='w')
        if sentaku == 0:
            wfile.write('Sphere\n')
        elif sentaku == 1:
            wfile.write('Rosenbrock\n')

        global L, L_half
        X = self.getRandomTeam()
        if L % 2 == 1:  #チーム数が奇数の場合、ダミーチームを作る
            DAMY_TEAM = [1000 for step in range(n)]
            X.append(DAMY_TEAM.copy())
            L += 1
            L_half += 1

        fX = list(0 for step in range(L))
        fX = self.optimizationFunction(sentaku, X, fX)  #適応度
        nextX = list(X)
        B = list(X)  #再良解
        fB = list(fX)  #最適な適応度
        f_best = min(fB)  #適応度の最も良い値
        f_bList = list()
        f_bList.append(f_best)

        t = 1
        schedule = self.leagueSchedule(t)  #schedule[週][チーム番号]
        wfile.write("t =%d, f_best=%f\n" % (t, f_best))
        while t < S * (L - 1):
            if L % 2 == 1:
                X.append(DAMY_TEAM.copy())
            if t % 10 == 0:
                wfile.write("t =%d, f_best=%f\n" % (t, f_best))
            Y = self.get_Y()
            for l in range(L - 1):
                teamA, teamB, teamC, teamD = self.teamClassification(
                    X, t, l - 1)
                winner1 = self.winORlose(X, teamA, teamB, fX, f_best)
                winner2 = self.winORlose(X, teamC, teamD, fX, f_best)
                nextX[X.index(teamA)] = self.setTeamFormation(
                    X, B, Y, teamA, teamB, teamC, teamD, winner1, winner2)

            X = nextX.copy()

            if L % 2 == 1:
                del X[-1]
            fX = self.optimizationFunction(sentaku, X, fX)  #適応度
            for l in range(L):
                if fX[l] < fB[l]:
                    B[l] = X[l]
                    fB[l] = fX[l]
            f_best = min(fB)
            f_bList.append(f_best)

            if t % (L - 1) == 0:
                #self.addOnModule() #add-onを追加できる
                schedule = self.leagueSchedule(t)
            t += 1

        wfile.write("~~~~出力結果~~~~\n")
        wfile.write("{x: new_x %f y: new_y %f}\n" % (fB[0], fB[1]))
        wfile.write("f(x,y) = %f\n" % f_best)
        self.shutsuryoku(fB, f_best)

        wfile.close()
        return f_bList

    def getRandomTeam(self):
        #チーム個体の初期値
        X = [[round(random.uniform(-10.0, 10.0), 6) for i in range(n)]
             for l in range(L)]  #6桁に丸める
        return X

    def leagueSchedule(self, t):
        #リーグスケジュールの設定
        if t == 1:
            schedule.append([l + 1 for l in range(L - 1)])
        randSche = random.sample([l + 1 for l in range(L - 1)], L - 1)
        schedule.append(randSche.copy())

        for l in range(L - 2):
            randSche.append(randSche.pop(0))
            schedule.append(randSche.copy())
        return schedule

    def teamClassification(self, X, t, l):
        """
        X[0]                   vs X[schedule[t][0]]
        X[schedule[t][L-2]]    vs X[schedule[t][1]]
        ...
        X[shcedule[t][L_half]] vs X[schedule[t][L_half-1]]
        """
        if l == -1:
            teamA = X[0]
            teamB = X[schedule[t][l + 1]]
            teamC = X[schedule[t][l + 2]]
            teamD = X[schedule[t][L - 1 - (l + 2)]]
        elif l == 0:
            teamA = X[schedule[t][l]]
            teamB = X[0]
            teamC = X[schedule[t][l + 2]]
            teamD = X[schedule[t][L - 1 - (l + 2)]]
        elif l == 1:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][L - 1 - l]]
            teamC = X[0]
            teamD = X[schedule[t][l - 1]]
        elif l == 2:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][L - 1 - l]]
            teamC = X[schedule[t][l - 2]]
            teamD = X[0]
        elif l < L_half:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][L - 1 - l]]
            teamC = X[schedule[t][L - l + 1]]
            teamD = X[schedule[t][l - 2]]
        elif l == L_half - 1:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][l + 1]]
            teamC = X[schedule[t][l + 3]]
            teamD = X[schedule[t][l - 2]]
        elif l == L_half:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][l - 1]]
            teamC = X[schedule[t][l + 1]]
            teamD = X[schedule[t][l - 2]]
        elif l == L_half + 1:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][l - 3]]
            teamC = X[schedule[t][l - 1]]
            teamD = X[schedule[t][l - 2]]
        elif l == L_half + 2:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][l - 5]]
            teamC = X[schedule[t][l - 3]]
            teamD = X[schedule[t][l - 2]]
        elif l < L - 1:
            teamA = X[schedule[t][l]]
            teamB = X[schedule[t][L - l - 1]]
            teamC = X[schedule[t][L - l + 1]]
            teamD = X[schedule[t][l - 2]]
        return teamA, teamB, teamC, teamD

    def optimizationFunction(self, sentaku, X, fX):
        """
        #HimmelblauFuncton
        #f(x,y) = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        fX[l] = (X[l][x]**2 + X[l][y] - 11)**2 + (
            X[l][x] + X[l][y]**2 - 7)**2
        """
        if sentaku is 0:
            #Sphere
            for l in range(L):
                fX[l] = 0
                #f(x) = \sigma{n}{i=1}x_i^2
                for i in range(n):
                    fX[l] += X[l][i]**2
            return fX
        elif sentaku is 1:
            #Rosenbrock
            for l in range(L):
                fX[l] = 0
                #f(x) = \sigma{n-1}{i=1}(100*(x_{i+1}-x_i)^2+(1-x_i)^2))
                for i in range(n - 1):
                    fX[l] += (
                        100 * (X[l][i + 1] - X[l][i])**2 + (1 - X[l][i])**2)

            return fX

    def winORlose(self, X, team1, team2, fX, f_best):
        Index1 = X.index(team1)
        Index2 = X.index(team2)
        winPoint = (fX[Index2] - f_best) / (
            fX[Index2] + fX[Index1] - 2.0 * f_best)

        shouritsu = random.uniform(0.0, 1.0)
        #勝敗
        if winPoint <= 0.5:
            if winPoint == 0.0:
                winner = team2
            elif shouritsu <= winPoint:
                winner = team1
            else:
                winner = team2
        else:
            if winPoint == 1.0:
                winner = team1
            elif winPoint <= shouritsu:
                winner = team2
            else:
                winner = team1
        return winner

    def get_Y(self):
        q0 = 1  #q_0=1   #フラグ本数
        Y = list()  #バイナリ変数配列
        y_sample = [i for i in range(n)]
        for i in range(L):
            y = [0 for num in range(n)]  #バイナリ変数
            a = random.uniform(0.0, 1.0)
            flagNum = (math.log(1 - (1 - (1 - p_c)**(n - q0 + 1)) * a) //
                       math.log(1 - p_c)) + q0 - 1
            q = int(flagNum)
            poInt = list(random.sample([l for l in range(n)], q))  #どれを1にするか
            poInt.sort()
            numy = 0
            numt = 0
            while numy < len(y) and numt < len(poInt):
                sa = y_sample[numy] - poInt[numt]
                if sa < 0:
                    numy += 1
                if sa == 0:
                    y[numy] = 1
                    numy += 1
                    numt += 1
                if sa > 0:
                    numt += 1
            Y.append(y)
        return Y

    def getRandom_rid(self):
        r_id1 = [random.uniform(0.0, 1.0) for i in range(n)]
        r_id2 = [random.uniform(0.0, 1.0) for i in range(n)]
        return r_id1, r_id2

    def addOnModule(self):
        return

    def setTeamFormation(self, X, B, Y, teamA, teamB, teamC, teamD, winner1,
                         winner2):
        r_id1, r_id2 = self.getRandom_rid()
        nextX = list(X)
        lA = X.index(teamA)
        lB = X.index(teamB)
        lC = X.index(teamC)
        lD = X.index(teamD)
        if winner1 == teamA and winner2 == teamC:  #S/T
            for i in range(n):
                nextX[lA][i] = B[lA][i] + Y[lA][i] * (
                    PSI1 * r_id1[i] * (X[lA][i] - X[lD][i]) + PSI1 * r_id2[i] *
                    (X[lA][i] - X[lB][i]))
        elif winner1 == teamA and winner2 == teamD:  #S/O
            for i in range(n):
                nextX[lA][i] = B[lA][i] + Y[lA][i] * (
                    PSI2 * r_id1[i] * (X[lD][i] - X[lA][i]) + PSI1 * r_id2[i] *
                    (X[lA][i] - X[lB][i]))
        elif winner1 == teamB and winner2 == teamC:  #W/T
            for i in range(n):
                nextX[lA][i] = B[lA][i] + Y[lA][i] * (
                    PSI1 * r_id1[i] * (X[lA][i] - X[lD][i]) + PSI2 * r_id2[i] *
                    (X[lB][i] - X[lA][i]))
        elif winner1 == teamB and winner2 == teamD:  #W/O
            for i in range(n):
                nextX[lA][i] = B[lA][i] + Y[lA][i] * (
                    PSI2 * r_id1[i] * (X[lD][i] - X[lA][i]) + PSI2 * r_id2[i] *
                    (X[lB][i] - X[lA][i]))
        return nextX[lA]

    def shutsuryoku(self, fB, f_best):
        print("~~~~出力結果~~~~")
        print("{x: new_x", fB[0], "y: new_y", fB[1], "}")
        print("f(x,y) = ", f_best)
        return


if __name__ == "__main__":
    print("[0]:Sphere")
    print("[1]:Rosenbrock")
    sentaku = int(input("ベンチマーク関数を選択:"))
    LCA = LeagueChampionshipAlgorithm()

    y = LCA.league(sentaku)

    x = list(t for t in range(S * (L - 1)))

    pf = plt.figure()
    plt.xlabel("Weeks")
    plt.ylabel("minima f(x)")
    plt.plot(x, y, label="LCA")

    date = datetime.datetime.now()
    if sentaku == 0:
        plt.title("LCA on Sphere")
        plt.show()
        pf.savefig('LCAPlot/LCA_Sphere{0:%Y%m%d_%H%M%S}.pdf'.format(date))
    elif sentaku == 1:
        plt.title("LCA on Rosenbrock")
        plt.show()
        pf.savefig('LCAPlot/LCA_Rosenbrock{0:%Y%m%d_%H%M%S}.pdf'.format(date))

    #QL.dumpQvalue()

    #for s in range(R.shape[0] - 1):
    #QL.runGreedy(s)
