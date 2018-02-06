# author: Franziska Meier
import numpy as np
from numpy import dot
from scipy.linalg import cholesky, inv


class LocalModel(object):

    def __init__(self, opt, D, K, lmD):
        self.K = K
        self.lmD = lmD
        self.D = D
        self.opt = opt

        self.set_initial_state()

    def set_initial_state(self):
        self.center = np.array(self.lmD)
        self.lengthscale = np.ones((1, self.lmD)) * self.opt.init_lambda

        self.muw = np.zeros((self.K, 1))
        self.Sigmaw = np.zeros((self.K, self.K))
        self.alpha_b_N = np.ones(self.K) * self.opt.alpha_b_0
        self.UsedK = np.arange(self.K, dtype=int)

        self.betaf_a_N = self.opt.betaf_a_0
        self.betaf_b_N = self.opt.betaf_b_0
        self.alpha_a_N = self.opt.alpha_a_0

        self.num_data = 0
        self.eta = self.opt.init_eta

        return

    def init_lm(self, c, X=None, Yh=None):
        self.center = c
        N = np.size(Yh)
        betaf = self.betaf_a_N / self.betaf_b_N
        alpha = self.alpha_a_N / self.alpha_b_N

        if (X is not None) and (Y is not None):
            w = self.get_activation(X)

            dist = X - c  # subtract center from each input data point
            Xh = np.zeros((N, self.K))
            Xh[:, 0:self.D] = w * dist
            Xh[:, -1] = w.squeeze()  # set bias term to 1.0

            if self.opt.var_approx_type == 1:
                SigmawI = np.dot(Xh.T, Xh) + np.diag(alpha)
            else:
                SigmawI = betaf * np.dot(Xh.T, Xh) + np.diag(alpha)

            L = cholesky(SigmawI, lower=True)
            LI = inv(L)
            self.Sigmaw = dot(LI.T, LI)

            if self.opt.var_approx_type == 1:
                self.muw = dot(dot(self.Sigmaw, Xh.T), Yh)
            else:
                self.muw = betaf * dot(dot(self.Sigmaw, Xh.T), Yh)

            return dot(Xh, self.muw)
        return

    def update(self, X, Y, Yh, w, s):
        # import ipdb
        # ipdb.set_trace()
        N = np.size(Y)
        Xh = self.center_and_prune_input(X)
        actK = np.size(self.UsedK)
        wXh = Xh * w
        # compute mean and var of hidden var f^n
        sigma = self.betaf_b_N / self.betaf_a_N
        sigmaf = sigma - (sigma ** 2) / s
        muf = np.dot(wXh, self.muw) + (1 / s) * sigma * Yh

        PhiPhi = np.dot(wXh.T, wXh)
        PhiF = dot(wXh.T, muf)
        muf2 = muf ** 2
        # update posterior over local regression parameters
        alpha = self.alpha_a_N / self.alpha_b_N
        betaf = self.betaf_a_N / self.betaf_b_N

        if self.opt.var_approx_type == 1:
            SigmawI = PhiPhi + np.diag(alpha)
        else:
            SigmawI = betaf * PhiPhi + np.diag(alpha)

        SigmawI = SigmawI + 1e-10 * np.eye(np.size(alpha))

        L = cholesky(SigmawI, lower=True)
        # TODO: can we prevent the inverse?
        LI = inv(L)
        self.Sigmaw = dot(LI.T, LI)

        if self.opt.var_approx_type == 1:
            self.muw = dot(self.Sigmaw, PhiF)
        else:
            self.muw = betaf * dot(self.Sigmaw, PhiF)

        self.betaf_a_N = self.opt.betaf_a_0 + 0.5 * (N + 1.0)

        # update posterior over precision parameters
        if self.opt.var_approx_type == 1:
            Nsigmaf = N * sigmaf
            self.betaf_b_N = self.opt.betaf_b_0 + 0.5 * (muf2.sum() - dot(dot(self.muw.T, SigmawI), self.muw) + N * sigmaf)
        else:
            muPhiPhimu = dot(self.muw.T, dot(PhiPhi, self.muw))
            sse = muf2.sum() - 2 * dot(PhiF.T, self.muw) + muPhiPhimu

            # TODO: check whether component wise multiplication is correct here
            tmp2 = (dot(PhiPhi, self.Sigmaw)).trace()

            Nsigmaf = N * sigmaf
            self.betaf_b_N = self.opt.betaf_b_0 + 0.5 * (sse + tmp2 + Nsigmaf)

        # TODO: check if algo is more robust if we change alpha updates
        self.alpha_a_N = self.opt.alpha_a_0 + 0.5
        dSigmaw = np.diag(self.Sigmaw)

        if self.opt.var_approx_type == 1:
            self.alpha_b_N = self.opt.alpha_a_0 + 0.5 * ( (self.betaf_a_N / self.betaf_a_N) * self.muw.T ** 2 + dSigmaw)
        else:
            self.alpha_b_N = self.opt.alpha_b_0 + 0.5 * (self.muw.T ** 2 + dSigmaw)

        # update length scales if we have seen enough data
        # TODO: can we replace the continuous lengthscale optimization through a discrete optimization?
        if self.opt.do_bwa:
            betaf = self.betaf_a_N / self.betaf_b_N

            dfx = self.lengthscale_gradient(X, wXh, muf, betaf)

            self.lengthscale = np.exp(np.log(self.lengthscale) - self.eta * dfx)

        return

    def predict(self, X):
        Xh = self.center_and_prune_input(X)
        return dot(Xh, self.muw)

    def center_and_prune_input(self, X):
        N = np.shape(X)[0]
        # centered data point, without bias element
        Xh = X - self.center
        nActK = np.size(self.UsedK)
        # if bias element is still active
        if self.UsedK[-1] == (self.K - 1):
            Xh = Xh[:, self.UsedK[0:(nActK - 1)] - 1]
            Xh = np.hstack((Xh, np.ones((N, 1))))
        else:
            Xh = Xh[:, self.UsedK - 1]

        return Xh

    def lengthscale_gradient(self, X, phi, muf, betaf):
        # N = np.shape(X)[0]
        sdist = (X - self.center) ** 2
        fp = dot(phi, self.muw)
        lengthscalesq = self.lengthscale ** 2

        if self.opt.var_approx_type == 1:
            E = dot(self.muw, self.muw.T) + self.get_variance()*self.Sigmaw
        else:
            E = dot(self.muw, self.muw.T) + self.Sigmaw

        phi_E = dot(phi, E)  # N x actK
        phi_E_phi = (phi_E * phi).sum(1, keepdims=True)  # N

        # ipdb.set_trace()
        sumvd = sdist * (phi_E_phi - fp * muf)  # N x D
        dfx = betaf * (sumvd.sum(0, keepdims=True) / lengthscalesq)  # D
        return dfx

    # def get_activation(self, x):
    #     sdist = (x - self.center) ** 2
    #     lengthscalesq = self.lengthscale ** 2
    #     return np.exp(-0.5 * (sdist / lengthscalesq).sum())
    def get_activation(self, X):
        # N = np.shape(X)[0]
        sdist = (X - self.center) ** 2
        lengthscalesq = self.lengthscale ** 2
        mdist = sdist / lengthscalesq
        return np.exp(-0.5 * np.sum(mdist, axis=1, keepdims=True))

    def get_variance(self):
        return self.betaf_b_N / self.betaf_a_N

    def get_alpha(self):
        return self.alpha_a_N / self.alpha_b_N

    def update_relevant_dimensions(self):
        nActK = np.size(self.UsedK)
        alpha = self.alpha_a_N / self.alpha_b_N

        alpha_upthresh = 999.999
        keep_idx = np.where(alpha[0] < alpha_upthresh)

        # check_idx = np.where(alpha[0] < 1e-10)
        # if np.size(check_idx) > 0:
        #     remove_idx = np.where(self.lengthscale[0] < 0.01)
        #     keep_idx = np.setdiff1d(keep_idx, remove_idx)

        new_size = np.size(keep_idx)
        if new_size == 0:
            return 0
        elif new_size < nActK:
            mask = np.zeros(nActK, dtype=bool)
            mask[keep_idx] = True
            self.alpha_b_N = self.alpha_b_N.T[mask].T
            self.UsedK = self.UsedK[mask]
            self.muw = self.muw[mask]

        return new_size

    def reset(self):
        self.set_initial_state()
        return

