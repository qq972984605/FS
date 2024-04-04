import numpy as np
from AFSIndividual import AFSIndividual
import random
import copy
import matplotlib.pyplot as plt



class ArtificialFishSwarm:
    """class for  ArtificialFishSwarm"""

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables, 2*vardim
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[visual, step, delta, trynum]
        '''

        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.lennorm = 3000
        self.Xichucun = np.zeros((self.MAXGEN, self.sizepop, self.vardim))

    def initialize(self):
        '''
        initialize the population of afs
        '''
        for i in range(0, self.sizepop):
            ind = AFSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self, x):
        '''
        evaluation the fitness of the individual
        '''
        x.calculateFitness()

    def forage(self, x):
        '''
        artificial fish foraging behavior
        '''
        newInd = copy.deepcopy(x)
        found = False
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])
            if indi.fitness > x.fitness:
                newInd.chrom = x.chrom + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                        indi.chrom - x.chrom) / np.linalg.norm(indi.chrom - x.chrom)
                newInd = indi
                found = True
                break
        if not (found):
            newInd = self.randSearch(x, self.params[1])
        return newInd

    def randSearch(self, x, searLen):
        '''
        artificial fish random search behavior
        '''
        ind = copy.deepcopy(x)
        ind.chrom += np.random.uniform(-1, 1,
                                       self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.chrom[j] < self.bound[0, j]:
                ind.chrom[j] = self.bound[0, j]
            if ind.chrom[j] > self.bound[1, j]:
                ind.chrom[j] = self.bound[1, j]
        self.evaluation(ind)
        return ind

    def huddle(self, x):
        '''
        artificial fish huddling behavior
        '''
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            xc = np.zeros(self.vardim)
            for i in range(0, nf):
                xc += self.population[index[i]].chrom
            xc = xc / nf
            cind = AFSIndividual(self.vardim, self.bound)
            cind.chrom = xc
            cind.calculateFitness()
            if (cind.fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "hudding"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def follow(self, x):
        '''
        artificial fish following behivior
        '''
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            best = -999999999.
            bestIndex = 0
            for i in range(0, nf):
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestIndex = index[i]
            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (
                                    self.population[bestIndex].chrom - x.chrom) / np.linalg.norm(
                    self.population[bestIndex].chrom - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "follow"
                if newInd.fitness > x.fitness:
                    return newInd
                else:
                    return self.forage(x)
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def solve(self):
        '''
        evolution process for afs algorithm
        '''
        t2 = 0
        of = 0
        j = 0
        l = 0
        m1 = 0
        m2 = 0
        m3 = 0
        t3 = []
        t1 = 1
        IF = 0
        op = 0
        temp5 = -99999999
        self.t = 0
        self.initialize()
        # evaluation the population
        for i in range(0, self.sizepop):
            self.evaluation(self.population[i])
            self.fitness[i] = self.population[i].fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        temp = copy.deepcopy(self.population[0])
        temp2 = copy.deepcopy(self.population[0])
        up = np.max(self.best.chrom)
        low = np.min(self.best.chrom)
        self.trace[self.t, 0] = -self.best.fitness
        self.trace[self.t, 1] = -self.avefitness
        temp3 = copy.deepcopy(self.population[bestIndex])
        print("Generation %d: optimal function value is: %6.30f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:

            self.t += 1
            # newpop = []

            temp1 = copy.deepcopy(self.population)

            for i in range(0, self.sizepop):
                xi1 = self.huddle(self.population[i])
                xi2 = self.follow(self.population[i])

                if xi1.fitness > xi2.fitness:
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                if xi2.fitness > xi1.fitness:
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness


            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = -self.best.fitness
            self.trace[self.t, 1] = -self.avefitness
            n = (11 * self.MAXGEN - 10 * self.t) * 100 / (self.MAXGEN * 11)
            cha = self.trace[self.t - 1, 0] - self.trace[self.t, 0]
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                l = 0

            if abs((self.trace[self.t, 0] - self.trace[self.t - 1, 0])) / self.trace[self.t, 0] > 0.005:
                t = 0
                j = 0
                m2 = 0
                t2 = 0
                l = 0
                t1 = 1
                m3 = 0
                m1 = 0
                op = 0

            else:
                l = l + 1
                m2 = m2 + 1
                m3 = m3 + 1

            if 0 < abs((self.trace[self.t, 0] - self.trace[self.t - 1, 0])) / self.trace[self.t, 0] < 0.005:
                op = 1
            if cha == 0:
                op = 0
                m1 = m1 + 1



            if l > 3 and cha == 0:
                t1 = 0.6
                t2 = t2 + 1


            if op == 1:
                self.params[0] = self.params[0] * (1 + 0.01 * j)
                self.params[1] = self.params[0] * 0.5
                j = j + 1

            if j <= 5 and op == 0:
                self.params[0] = self.params[0] * (n / (n + 1)) * t1
                self.params[1] = self.params[0] * 0.75
                j = j + 1

            worstIndex = np.argmin(self.fitness)

            if m2 > 50:
                for a in range(0, self.sizepop - 1):
                    m11 = copy.deepcopy(self.population[bestIndex])
                    for b in range(0, self.vardim):
                        m11 = copy.deepcopy(self.population[bestIndex])
                        m11.chrom[b] = self.population[a].chrom[b]
                        self.evaluation(m11)
                        if m11.fitness > self.population[bestIndex].fitness:
                            self.population[bestIndex] = m11

                m2 = 0
                op = 0

            temp4 = copy.deepcopy(self.population[bestIndex])
            if cha > 0:
                temp4.chrom = self.population[bestIndex].chrom + (self.population[bestIndex].chrom - temp3.chrom)
                self.evaluation(temp4)
                if self.population[bestIndex].fitness < temp4.fitness:
                    self.population[bestIndex] = temp4
                    self.fitness[bestIndex] = temp4.fitness
            temp3 = copy.deepcopy(self.population[bestIndex])

            if self.population[bestIndex].fitness > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                l = 0

            if m3 > 100:
                for a in range(0, self.vardim):
                    temp2 = copy.deepcopy(temp)
                    m11 = copy.deepcopy(self.best)
                    temp2.chrom[a] = m11.chrom[a]
                    self.evaluation(temp2)
                    if m11.fitness > temp.fitness:
                        temp.chrom[a] = m11.chrom[a]
                        self.evaluation(temp)
                self.best = temp

            up = np.max(self.best.chrom)
            low = np.min(self.best.chrom)

            if abs(up) > abs(low):
                if low >=0:
                    bound1 = -low
                    bound2 = up
                if low <0:
                    bound1 = low
                    bound2 = up
            if abs(low) >= abs(up):
                if up >=0:
                    bound1 = low
                    bound2 = up
                if up < 0:
                    bound1 = low
                    bound2 = -up


            if m3 > 100:
                for a in range(0, self.sizepop):
                    for b in range(0, self.vardim):
                        self.bound[0, b] = bound1 - (up - low) / 10
                        self.bound[1, b] = bound2 + (up - low) / 10
                        if random.random() > 0.5:
                            self.population[a].chrom[b] = random.uniform(
                                self.best.chrom[b] + (self.bound[1, b] - self.bound[0, b]) / 2, self.bound[1, b])
                        else:
                            self.population[a].chrom[b] = random.uniform(self.bound[0, b], self.best.chrom[b] - (
                                        self.bound[1, b] - self.bound[0, b]) / 2)
                        if self.population[a].chrom[b] < self.bound[0, b]:
                            self.population[a].chrom[b] = random.uniform(
                                self.best.chrom[b] + (self.bound[1, b] - self.bound[0, b]) / 2, self.bound[1, b])
                        if self.population[a].chrom[b] > self.bound[1, b]:
                            self.population[a].chrom[b] = random.uniform(self.bound[0, b], self.best.chrom[b] - (
                                        self.bound[1, b] - self.bound[0, b]) / 2)
                        self.evaluation(self.population[a])
                        bestIndex = np.argmax(self.fitness)
                self.params[0] = visual * (up - low) * 1.2
                self.params[1] = step * (up - low) * 1.2
                temp = self.best
                self.best.fitness = -9999.
                self.best = copy.deepcopy(self.population[bestIndex])
                m3 = 0
                of = 1


            if l > 7:
                l = 0
                j = 0

            if temp5 < self.best.fitness:
                temp5 = self.best.fitness


            print("Generation %d: optimal function value is: %6.30f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

            print(self.params[0])
            print(self.best.fitness)
            print(self.best.chrom)
            print(-temp5)

        print("Optimal function value is: %6.30f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.trace[self.t, 0])
        print(self.best.chrom)
        self.printResult()

    def distance(self, x):
        '''
        return the distance array to a individual
        '''
        dist = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            dist[i] = np.linalg.norm(x.chrom - self.population[i].chrom) / self.lennorm
        return dist

    def printResult(self):
        '''
        plot the result of afs algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.semilogy(x, y1, 'r', label='optimal value')
        plt.semilogy(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Rastrigin")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    boundall1 = 100
    boundall2 = 0
    visual = (boundall2 - boundall1) * 1.5 /3000
    step = visual * 0.75
    bound = np.tile([[boundall1], [boundall2]], 2)
    afs = ArtificialFishSwarm(5, 2, bound, 5, [visual, step, 0.75, 5])
    afs.solve()

