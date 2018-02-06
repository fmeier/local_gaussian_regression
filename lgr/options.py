# author: Franziska Meier
class Options(object):
    '''parameter settings for LGR'''

    def __init__(self, lmD):
        ''' setting default params '''
        self.max_iter = 100
        self.init_lambda = 0.3
        self.activ_thresh = 0.5
        self.init_eta = 0.0001
        self.fr = 0.999
        self.norm_out = 1.0
        self.max_num_lm = 1000
        self.alpha_a_0 = 1e-6
        self.alpha_b_0 = 1e-6
        self.betaf_a_0 = 1e-6
        self.betaf_b_0 = 1e-6

        self.betay = 1e9
        self.lmD = lmD
        self.do_bwa = True  # do lenghtscale optimization
        self.do_pruning = True

        self.var_approx_type = 0  # 0: fully factorized, 1: w,beta one factor

    def print_options(self):
        print "options: "
        print " norm_out: " + str(self.norm_out)
        print " max_iter: " + str(self.max_iter)
        print " init_lambda: " + str(self.init_lambda)
        print " activ thresh: " + str(self.activ_thresh)
