from surprise import AlgoBase
from surprise import Dataset
from surprise import PredictionImpossible
from surprise.model_selection import cross_validate
import numpy as np

class KNNSP(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]
        self.ii_sim_dict = {}
        for iiid1 in trainset.all_items():
            self.ii_sim_dict[iiid1] = {}
            for iiid2 in trainset.all_items():
                self.ii_sim_dict[iiid1][iiid2] = self.sim_between_movies(iiid1, iiid2)
        return self


    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        ir_dict = dict(self.trainset.ur[u])
        neighbors = [(k, self.ii_sim_dict[i][k]) for (k, r) in self.trainset.ur[u]]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        top = 0
        bot = 0
        for iiid, r in neighbors[:min(len(neighbors), 5)]:
            top += r * ir_dict.get(iiid)
            bot += r
        if bot == 0:
            return self.user_mean[u]
        return top/bot


    def sim_between_movies(self, iiid1, iiid2):
        comm = 0
        if iiid1 == iiid2:
            return 1
        movieARating = dict(self.trainset.ir[iiid1])
        movieBRating = dict(self.trainset.ir[iiid2])
        set1 = set(movieARating.keys())
        set2 = set(movieBRating.keys())
        factor = len(set1.intersection(set2))
        for k in set1.intersection(set2):
            v = movieARating.get(k)
            v2 = movieBRating.get(k)
            comm += (1 - abs(v2 - v)/factor)
        return comm/len(set1.union(set2))
