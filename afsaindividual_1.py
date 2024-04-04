import numpy as np
import ObjFunction
import random

class afsaindividual_1:

    def __init__(self, bound):
        '''
        bound: boundaries of variables
        '''
        self.bound = bound

    def generate(self, x):
        '''
        generate a rondom chromsome
        '''
        len = np.random.randint(self.bound[0], x)
        datalen = list(range(self.bound[1]))
        self.chrom = random.sample(datalen, len)
        self.binary = np.zeros(self.bound[1])
        for i in self.chrom:
            self.binary[i] = 1
        self.feature = len
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = ObjFunction.m(
            self.chrom, self.bound)
