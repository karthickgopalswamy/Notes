
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import datetime
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt


class Stock_data:
    def __init__(self, start_date, end_date, period,risk_factor):
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        temp = data[0]
        self.tickers = temp['Symbol'].to_list()
        self.risk_factor = risk_factor
        self.start = start_date
        self.end = end_date
        self.sp_500 = pd.DataFrame(columns=['Return', 'Risk'], index=self.tickers)
        self.sp_dict = {}
        self.sp_cov = None
        self.Sp_max = None
        self.no_stock = None
        self.period = period
        self.names = None

    def get_stock(self,name):
        try:
            prices = web.DataReader(name, 'yahoo', self.start, self.end)
        except:
            return None
        Endings = prices.Close
        for i in range(len(Endings)):
            if Endings[i] is np.nan:
                Endings[i] = Endings[i - 1]
        increases = []
        for i in range(len(Endings) - self.period):
            increase = (Endings[i + self.period] - Endings[i]) / Endings[i]
            increases.append(increase)
        increases_ = np.array(increases)
        return (np.mean(increases_), np.std(increases_))

    def scrap_data(self):
        for name in self.tickers:
            new_stock = self.get_stock(name=name)
            if new_stock is not None:
                self.sp_500.loc[name] = [new_stock[0], new_stock[1]]
        print(self.sp_500.head())

        plt.plot(self.sp_500.Risk.values, self.sp_500.Return.values, 'r.')
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.plot(np.arange(0, 0.18, 0.01), np.arange(0, 0.18, 0.01) * self.risk_factor, 'g--')
        plt.show()
        self.Sp_max = self.sp_500[self.sp_500.Return > self.sp_500.Risk * self.risk_factor]
        self.no_stock = self.Sp_max.shape[0]
        self.names = self.Sp_max.index.to_list()

    def calculate_increases(self,name):
        try:
            prices = web.DataReader(name, 'yahoo', self.start, self.end)
        except:
            return None

        Endings = prices.Close
        for i in range(len(Endings)):
            if Endings[i] is np.nan: Endings[i] = Endings[i - 1]
        increases = []
        for i in range(len(Endings) - self.period):
            increase = (Endings[i + self.period] - Endings[i]) / Endings[i]
            increases.append(increase)
        increases = np.array(increases)
        return increases

    def get_covariances(self):
        self.sp_cov = {}

        for name in self.names:
            self.sp_dict[name] = self.calculate_increases(name)

        for i in range(self.no_stock):
            for j in range(self. no_stock):
                cov = np.cov(self.sp_dict[self.names[i]], self.sp_dict[self.names[j]])[0][1]
                self.sp_cov[(self.names[i], self.names[j])] = cov
        print(self.sp_cov)

def main():
    SD = Stock_data(start_date = datetime.datetime(2018,10,1),end_date = datetime.datetime(2019,10,1),period = 20,risk_factor=0.5)
    SD.scrap_data()
    SD.get_covariances()

    opt = SolverFactory('cplex', executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')

    def sumWeights(model):
        return sum(model.weight[name] for name in model.Stock) == 1

    def ExpectedReturn(model):
        return sum(model.weight[name] * model.Return[name] for name in model.Stock) >= model.ReturnThreshold

    def Risk(model):
        return sum(model.Covariance[(i,j)]*model.weight[i]*model.weight[j] for i in model.Stock for j in model.Stock)

    model = AbstractModel()
    model.Stock = Set()
    model.Return = Param(model.Stock,default=0)
    model.Covariance = Param(model.Stock*model.Stock,default=0)
    model.ReturnThreshold = Param(within=NonNegativeReals,default=0.05,mutable=True)
    model.weight = Var(model.Stock,domain=NonNegativeReals, bounds=(0, 1))
    model.WeightSum = Constraint(rule=sumWeights)
    model.ExpectedReturn = Constraint(rule=ExpectedReturn)
    model.Risk = Objective(rule=Risk, sense=minimize)


    data = {None: {
        'Stock': {None: SD.names},
        'Return': SD.Sp_max['Return'].to_dict(),
        'Covariance': SD.sp_cov,
        }
    }

    instance = model.create_instance(data)

#
    result = opt.solve(instance)
    solution = {}
    for stock in instance.Stock:
        sol = round(value(instance.weight[stock]),3)
        if sol != 0.0:
            solution[stock] = sol
    soldf = pd.DataFrame.from_dict(solution)
    soldf.rename_axis(index = {0: "Weight"},inplace=True)
    soldf.sort_values(axis=1,by=['Weight'],ascending=False)
    print(soldf)
    print(value(instance.Risk))

#
#
#
#
#
# if __name__ == "__main__": main()
