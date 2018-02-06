# author: Franziska Meier
# implements incremental LGR

import numpy as np
from localmodel import LocalModel



class LGR(object):
    ''' (batch) Local Gaussian Regression'''

    def __init__(self, opt, dim):
        #opt.print_options()

        self.D = dim  # dim of data
        self.K = dim + 1  # number of dim of each local model
        self.lmD = opt.lmD  # number of dim of each localizer
        self.M = 0  # number of local models
        self.opt = opt
        self.betay = opt.betay

        self.lmodels = [None] * opt.max_num_lm
        for i in range(0, opt.max_num_lm):
            self.lmodels[i] = LocalModel(opt, dim, self.K, self.lmD)

    def add_local_model(self, x, X=None, Yh=None):

        if(self.M + 1 < self.opt.max_num_lm):
            self.lmodels[self.M].init_lm(x, X, Yh)
            self.M = self.M + 1
        else:
            print "maximum number of local models reached"

        return 0

    def update(self, X, Y):

        yp = 0.0
        lm_var = np.zeros(self.M)
        for m in range(0, self.M):
            lm = self.lmodels[m]
            lm_var[m] = lm.get_variance()
            wm = lm.get_activation(X)
            yp += wm * lm.predict(X)

        yh = Y - yp
        s = 1.0 / self.betay + lm_var.sum()  # total amount of variance
        # ipdb.set_trace()
        for m in range(0, self.M):
            lm = self.lmodels[m]
            wm = lm.get_activation(X)
            lm.update(X, Y, yh, wm, s)

        # pruning
        if self.opt.do_pruning:
            m = 0
            while m < self.M:
                rel_dim = self.lmodels[m].update_relevant_dimensions()
                if rel_dim == 0:
                    # swap the pruned out model with the last model
                    self.lmodels[m], self.lmodels[self.M - 1] = self.lmodels[self.M - 1], self.lmodels[m]

                    # reset the pruned out model which is now the last one
                    self.lmodels[self.M - 1].reset()
                    self.M = self.M - 1
                    m -= 1

                m += 1

        if self.lmodels[self.M].UsedK.size < 2:
            dum = 0

        # return prediction before new local model was added
        return yp

    def initialize_local_models(self, X):
        n_data = X.shape[0]

        self.add_local_model(X[0, :])

        for n in range(0, n_data):
            xn = X[n, :]
            w = np.zeros(self.M)
            for m in range(0, self.M):
                lm = self.lmodels[m]
                w[m] = lm.get_activation(xn[np.newaxis, :])
            
            max_act = w.max()
            if max_act < self.opt.activ_thresh:
                self.add_local_model(xn)

    def run(self, X, Y, n_iter, debug):

        n_data = np.size(Y)
        Yp = self.predict(X)
        sse = ((Yp - Y) ** 2).sum()
        mse = sse / n_data
        print "initial nmse: " + str(mse/np.var(Y))
        nmse = np.zeros(n_iter)

        # learn parameters
        for i in range(0, n_iter):

            sse = 0.0
            # batch update parameters
            self.update(X, Y)

            Yp = self.predict(X)
            sse = sse + ((Y - Yp) ** 2).sum()
            mse = sse / n_data
            nmse[i] = mse / np.var(Y)

            # compute current mse
            if debug and i > 0 and np.mod(i, 100) == 0:
                print "iter: {}, nmse: {}, M: {}".format(i, nmse[i], self.M)

        # models final prediction on training data
        # Yp = self.predict(X)
        # sse = sse + ((Y - Yp) ** 2).sum()
        # mse = sse / n_data
        # nmse = mse / np.var(Y)
        return nmse

    def predict(self, x):

        yp = 0.0
        for m in range(0, self.M):
            w = self.lmodels[m].get_activation(x)
            yp = yp + w * self.lmodels[m].predict(x)

        return yp


    def get_local_model_activations(self, X):

        local_models_act = np.zeros((X.shape[0], self.M))
        for m in range(self.M):
            local_models_act[:, m] = self.lmodels[m].get_activation(X)[:, 0]

        return local_models_act
            
