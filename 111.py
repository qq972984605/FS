import numpy as np
from afsaindividual_1 import afsaindividual_1
import random
import copy
import matplotlib.pyplot as plt
import heapq
import warnings
warnings.filterwarnings("ignore")


class ArtificialFishSwarm:
    """class for  ArtificialFishSwarm"""

    def __init__(self, sizepop, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables, 2*vardim
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[visual, step, 拥挤因子, trynum, library]
        '''


        self.sizepop = sizepop
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))


    def initialize(self, x):
        '''
        initialize the population of afs
        '''
        for i in range(0, self.sizepop):
            ind = afsaindividual_1(self.bound)
            ind.generate(x)
            self.population.append(ind)


    def evaluation(self, x):
        '''
        evaluation the fitness of the individual
        '''
        x.calculateFitness()


    def distance(self, x):
        dist = np.zeros(self.sizepop)
        count = 0
        for i in range(self.sizepop):
            d = x.binary - self.population[i].binary
            for j in d:
                if j != 0:
                    count += 1
            dist[i] = count
            count = 0
        return dist


    def huddle(self, x, y, z):
        ind = copy.deepcopy(x)
        index = []
        dist = self.distance(x)
        center = []
        chaset = []
        indchrom = []
        for i in range(self.sizepop):
            if dist[i] > 0 and dist[i] <= self.params[0]:
                index.append(i)
        nf = len(index)
        if nf > 0:
            xc = np.zeros(self.bound[1])
            for i in range(nf):
                xc += self.population[index[i]].binary
            xc = xc / nf
            cind = copy.deepcopy(ind)
            for i in range(len(xc)):
                if xc[i] < 0.5:
                    xc[i] = 0
                else:
                    xc[i] = 1
            cind.binary = xc
            xnext = np.array(ind.binary)
            cha = xnext - xc
            for i in range(len(xc)):
                if xc[i] != 0:
                    center.append(i)
                    cind.chrom = center
            cind = copy.deepcopy(ind)
            if cind.fitness / nf > x.fitness * self.params[2]:
                for i in range(len(cha)):
                    if cha[i] != 0:
                        chaset.append(i)
                step = int(random.uniform(0, 1) * self.params[1]) + 1
                if step <= len(chaset):
                    add = random.sample(chaset, step)
                    for i in add:
                        xnext[i] = xnext[i] - cha[i]
                else:
                    add = random.sample(range(len(ind.binary)), step - len(chaset))
                    for i in add:
                        xc[i] = abs(xc[i] - 1)
                    xnext = xc
                ind.binary = xnext
                for i in range(len(ind.binary)):
                    if ind.binary[i] != 0:
                        indchrom.append(i)
                ind.chrom = indchrom
                ind.feature = len(ind.chrom)
                self.evaluation(ind)
                return (ind)
            else:
                return (self.forage(x, y, z))
        else:
            return (self.forage(x, y, z))


    def follow(self,x, y, z):
        ind = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        chaset = []
        indchrom = []
        for i in range(self.sizepop):
            if dist[i] > 0 and dist[i] <= self.params[0]:
                index.append(i)
        nf = len(index)
        if nf > 0:
            best = -999999999.
            for i in range(nf):
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestchrom = len(self.population[index[i]].chrom)
                    bestIndex = index[i]
                if self.population[index[i]].fitness == best:
                    if len(self.population[index[i]].chrom) < bestchrom:
                        best = self.population[index[i]].fitness
                        bestchrom = len(self.population[index[i]].chrom)
                        bestIndex = index[i]
                if self.population[bestIndex].fitness / nf > x.fitness * self.params[2]:
                    bestbinary = np.array(self.population[bestIndex].binary)
                    bestlist = list(bestbinary)
                    xnext = np.array(ind.binary)
                    cha = xnext - bestbinary
                    for i in range(len(cha)):
                        if cha[i] != 0:
                            chaset.append(i)
                    step = int(random.uniform(0, 1) * self.params[1]) + 1
                    if step <= len(chaset):
                        add = random.sample(chaset, step)
                        for i in add:
                            xnext[i] = xnext[i] - cha[i]
                    else:
                        add = random.sample(range(self.bound[1]), step - len(chaset))
                        for i in add:
                            bestlist[i] = abs(bestlist[i] - 1)
                            xnext = bestlist
                    xnext = np.array(xnext)
                    ind.binary = xnext
                    for i in range(len(ind.binary)):
                        if ind.binary[i] != 0:
                            indchrom.append(i)
                    ind.chrom = indchrom
                    ind.feature = len(ind.chrom)
                    self.evaluation(ind)
                    return (ind)
            else:
                return (self.forage(x, y, z))
        else:
            return (self.forage(x, y, z))


    def forage(self, x, y, z):
        newind = copy.deepcopy(x)
        found = False
        for i in range(self.params[3]):
            indi = self.randSearch(x, int(random.uniform(0,1) * self.params[0]), y, z)
            if indi.fitness > x.fitness:
                newind = indi
                found = True
                break
            if indi.fitness == x.fitness:
                if indi.feature < x.feature:
                    newind = indi
                    found = True
                    break
        if not (found):
            newind = self.randSearch(x, int(random.uniform(0,1) * self.params[0]), y, z)
        return newind


    def randSearch(self, x, searlen, y, z):

        ind = copy.deepcopy(x)
        indchrom = []
        xnext = ind.binary

        if z == 0:
            add = random.sample(y, searlen)

            xnext = np.array(xnext)
            for i in add:
                xnext[i] = abs(xnext[i] - 1)
            ind.binary = xnext

        if z == 1:
            if searlen < len(ind.chrom):
                red = random.sample(ind.chrom, searlen)
            if searlen >= len(ind.chrom):
                if len(ind.chrom) <= 2:
                    red = []
                else:
                    red = random.sample(ind.chrom, 1)
            for i in red:
                ind.binary[i] = 0

        for i in range(len(xnext)):
            if xnext[i] != 0:
                indchrom.append(i)

        ind.chrom = indchrom
        ind.feature = len(ind.chrom)
        self.evaluation(ind)
        return ind



    def library(self, x, y):
        ind = copy.deepcopy(x)
        a = []
        for i in range(len(ind.binary)):
            if ind.binary[i] == 0:
                a.append(i)
        if len(a) < y:
            y = len(a)
        c = random.sample(a, y)
        return c

    def reset(self, x, y):
        ind = copy.deepcopy(x)
        c = copy.deepcopy(y)
        for i in c:
            ind.binary[i] = 1
            ind.chrom.append(i)
        self.evaluation(ind)
        return ind



    def solve(self):
        self.t = 0
        a = 0
        b = 0
        feature = 99999
        self.initialize(self.bound[1])
        for i in range(self.sizepop):
            self.evaluation(self.population[i])
            self.fitness[i] = self.population[i].fitness

        best = np.max(self.fitness)
        b = np.where(self.fitness == best)
        for i in list(b[0]):
            if self.population[i].feature < feature:
                bestindex = i
                feature = self.population[i].feature
        self.best = copy.deepcopy(self.population[bestindex])
        self.trace[0, 0] = self.best.fitness
        self.trace[0, 1] = len(self.best.chrom)
        temp = 0
        temp1 = 0
        alllibrary = list(range(self.bound[1]))
        temp0 = 0
        t1 = copy.deepcopy(self.population[0])
        z = 0

        #print(self.best.chrom)

        while self.t < self.MAXGEN - 1:
            self.t += 1

            if t1.fitness < self.best.fitness:
                t1 = copy.deepcopy(self.best)
            if t1.fitness == self.best.fitness:
                if len(t1.chrom) > len(self.best.chrom):
                    t1 = copy.deepcopy(self.best)

            print(self.t)
            print(self.best.chrom)
            print(self.best.fitness)
            print(len(self.best.chrom))


            for i in range(self.sizepop):
                xi1 = self.huddle(self.population[i], alllibrary, z)
                xi2 = self.follow(self.population[i], alllibrary, z)
                if self.population[i].fitness <= xi1.fitness or self.population[i].fitness <= xi2.fitness:
                    if xi1.fitness > xi2.fitness:
                        self.population[i] = xi1
                        self.fitness[i] = xi1.fitness

                    if xi2.fitness > xi1.fitness:
                        self.population[i] = xi2
                        self.fitness[i] = xi2.fitness

                    if xi1.fitness == xi2.fitness:
                        if xi1.feature < xi2.feature:
                            self.population[i] = xi1
                            self.fitness[i] = xi1.fitness

                        else:
                            self.population[i] = xi2
                            self.fitness[i] = xi2.fitness

            z = 0
            best = np.max(self.fitness)
            bestindex = np.argmax(self.fitness)
            if self.population[bestindex].fitness > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestindex])
                temp = 0
                temp1 = 0
            if self.population[bestindex].fitness == self.best.fitness:
                if len(self.population[bestindex].chrom) < len(self.best.chrom):
                    self.best = copy.deepcopy(self.population[bestindex])
                    temp = 0
                    temp1 = 0


            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = len(self.best.chrom)



            temp0 = self.best.fitness

        print(t1.fitness)
        print(t1.chrom)
        self.printResult()
        self.printresult()

    def printResult(self):
        '''
        plot the result of afs algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.semilogy(x, y1, 'r')
        plt.xlabel("Number of Iteration")
        plt.ylabel("Accuracy")
        plt.title("Lung")
        plt.legend()
        plt.show()

    def printresult(self):
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y2, 'b')
        plt.xlabel("Number of Iteration")
        plt.ylabel("Number of Feature")
        plt.title("Lung")
        plt.legend()
        plt.show()



if __name__ == "__main__":
    boundall1 = 2
    boundall2 = 200
    bound = [boundall1, boundall2]
    visual = 50
    step = 25
    for i in range(10):
        afs = ArtificialFishSwarm(30, bound, 500, [visual, step, 0.75, 5, 3])
        afs.solve()
