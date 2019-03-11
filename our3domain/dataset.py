from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

class Result():
    def get_imxz(self):
        return self.imxz
    def get_imyz(self):
        return self.imyz
    def get_imwz(self):
        return self.imwz
    def get_imxzy(self):
        return self.imxzy
    def get_imyzw(self):
        return self.imyzw
    def get_imwzx(self):
        return self.imwzx
    def get_rmxzyz(self):
        return self.rmxzyz
    def get_rmyzwz(self):
        return self.rmyzwz
    def get_rmwzxz(self):
        return self.rmwzxz
    def get_rmxzyzx(self):
        return self.rmxzyzx
    def get_rmyzy(self):
        return self.rmyzy
    def get_rmyzwzy(self):
        return self.rmyzwzy
    def get_rmxzx(self):
        return self.rmxzx
    def get_rmwzxzw(self):
        return self.rmwzxzw
    def get_rmwzw(self):
        return self.rmwzw
    def set_z_domian(self, imzx, imzy, imzw, rmzxz, rmzyz, rmzwz):
        self.imzx = imzx
        self.imzy = imzy
        self.imzw = imzw 
        self.rmzxz = rmzxz
        self.rmzyz = rmzyz
        self.rmzwz = rmzwz
    def set_all(self, imxz, imyz, imwz, imxzy, imyzw, imwzx, rmxzyz, rmyzwz, rmwzxz, rmxzyzx, rmyzy, rmyzwzy, rmxzx, rmwzxzw, rmwzw):
        self.imxz = imxz
        self.imyz = imyz
        self.imwz = imwz
        self.imxzy = imxzy
        self.imyzw = imyzw
        self.imwzx = imwzx
        self.rmxzyz = rmxzyz
        self.rmyzwz = rmyzwz
        self.rmwzxz = rmwzxz
        self.rmxzyzx = rmxzyzx
        self.rmyzy = rmyzy
        self.rmyzwzy = rmyzwzy
        self.rmxzx = rmxzx
        self.rmwzxzw = rmwzxzw
        self.rmwzw = rmwzw
    def get_all(self):
        return self.imxz,self.imyz,self.imwz,self.imxzy,self.imyzw,self.imwzx,self.rmxzyz,self.rmyzwz,self.rmwzxz,self.rmxzyzx,self.rmyzy,self.rmyzwzy,self.rmxzx,self.rmwzxzw,self.rmwzw
    def get_z_domain(self):
        return self.imzx,self.imzy,self.imzw, self.rmzxz, self.rmzyz, self.rmzwz

class BaseData():

    def set_means(self, means_data):
        self.means_data = means_data
    
    def set_variances(self, variances):
        self.variances = variances

    def set_priors(self, priors):
        self.priors = priors

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_dataset_test(self, dataset_test):
        self.dataset_test = dataset_test

    def set_label_test(self, label_test):
        self.label_test = label_test

    def get_means(self):
        return self.means_data
    
    def get_variances(self):
        return self.variances
    
    def get_priors(self):
        return self.priors

    def get_dataset(self):
        return self.dataset

    def get_dataset_test(self):
        return self.dataset_test
    
    def get_label_test(self):
        return self.label_test

    def create_dataset(self, opt, means_x, std_x, name):
        means_x = list(means_x)
        variances_x = [np.eye(2) * std_x for _ in means_x]
        priors_x = [1.0/len(means_x) for _ in means_x]
       
        dataset_x = sample_GMM(opt.dataset_size_x, means_x, variances_x, priors_x, sources=('features', ))
        save_path_x = opt.result_dir + name
    
        # plot_GMM(dataset, save_path)
        ##  reconstruced x

        X_dataset  = dataset_x.data['samples']
        X_targets = dataset_x.data['label']

        self.set_means(means_x)
        self.set_priors(priors_x)
        self.set_variances(variances_x)
        self.set_dataset(X_dataset)

        fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c=cm.Set1(X_targets.astype(float)/opt.input_dim/2.0),
           edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.axis('on')
        plt.savefig(save_path_x, transparent=True, bbox_inches='tight')
    
    def create_dataset_test(self, opt, std_x, name):
        variances_x = self.get_variances()
        priors_x = self.get_priors()
        means_x = self.get_means()
        # create X dataset
        datasetX_test = sample_GMM(opt.dataset_size_x_test, means_x, variances_x, priors_x, sources=('features', ))
        save_path = opt.result_dir + name
        
        # plot_GMM(dataset, save_path)
        ##  reconstruced x
        
        X_np_data_test = datasetX_test.data['samples']
        X_targets_test = datasetX_test.data['label']

        self.set_dataset_test(X_np_data_test)
        self.set_label_test(X_targets_test)

        fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ax.scatter(X_np_data_test[:, 0], X_np_data_test[:, 1], c=cm.Set1(X_targets_test.astype(float)/opt.input_dim/2.0),
           edgecolor='none', alpha=0.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.axis('on')
        plt.savefig(save_path, transparent=True, bbox_inches='tight')
        
    def init_dataset(self,opt,mean_x, std_x, x_train, x_test):
        self.create_dataset(opt, mean_x, std_x, x_train )
        self.create_dataset_test(opt, std_x, x_test)
    

def init_datasets(opt):
    mean_w = map(lambda w:  np.array(w), [[2, 0],
                                     [-2, 0],
                                     [0, 0],
                                     [0, 2],
                                     [0, -2]])

    mean_x = map(lambda x:  np.array(x),[[0, 0],
                                     [2, 2],
                                     [-2, -2],
                                     [2, -2],
                                     [-2, 2]
                                     ])

    mean_y = map(lambda y:  np.array(y),[[0, 2],
                                     [2, 0.547],
                                     [-2, 0.547],
                                     [1.236, -2],
                                     [-1.236, -2]
                                     ])

    mean_z = map(lambda z:  np.array(z),[[0, 0]])
    
    """ Create dataset """
    data_x = BaseData()
    data_y = BaseData()
    data_w = BaseData()
    data_z = BaseData()
    data_x.init_dataset(opt, mean_x, opt.std_x, opt.x_train, opt.x_test)
    data_y.init_dataset(opt, mean_y, opt.std_y, opt.y_train, opt.y_test)
    data_w.init_dataset(opt, mean_w, opt.std_w, opt.w_train, opt.w_test)
    data_z.init_dataset(opt, mean_z, opt.std_z, opt.z_train, opt.z_test)
    return data_x, data_y, data_w, data_z

def plot(opt,datasets, n_viz,imz, name,label_x, label_y):
    ## inferred marginal z
    fig_mz, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    ll = np.tile(datasets, (n_viz))
    ax.scatter(imz[:, 0], imz[:, 1], c=cm.Set1(ll.astype(float)/opt.input_dim/2.0),
        edgecolor='none', alpha=0.5)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    ax.axis('on')
    plt.savefig(opt.result_dir + name, transparent=True, bbox_inches='tight')

def plot_lr(opt, name, FD, FG, label_1, label_2):
    ## learning curves
    fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    ax.plot(FD, label="Discriminator")
    ax.plot(np.array(FG)[:,0], label="Generator")
    ax.plot(np.array(FG)[:,1], label=label_1)
    ax.plot(np.array(FG)[:,2], label=label_2)
    plt.xlabel('Iteration')
    plt.xlabel('Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.axis('on')
    plt.savefig(opt.result_dir + name, bbox_inches='tight')

def print_xz(epoch, i, f_d, f_g0, f_g1, f_g2, f_g3):
    print("epoch %d iter %d: discloss %f genloss %f adv_x %f recons_x %f recons_z %f" % (epoch, i, f_d, f_g0, f_g1, f_g2, f_g3))

def print_yz(epoch, i, f_d, f_g0, f_g1, f_g2, f_g3):
    print("epoch %d iter %d: discloss %f genloss %f adv_y %f recons_y %f recons_z %f" % (epoch, i, f_d, f_g0, f_g1, f_g2, f_g3))

def print_wz(epoch, i, f_d, f_g0, f_g1, f_g2, f_g3):
    print("epoch %d iter %d: discloss %f genloss %f adv_w %f recons_w %f recons_z %f" % (epoch, i, f_d, f_g0, f_g1, f_g2, f_g3))

def print_xy(epoch, i, f_d, f_g0, f_g1, f_g2, f_g3):
    print("epoch %d iter %d: discloss %f genloss %f adv_x %f recons_x %f recons_y %f" % (epoch, i, f_d, f_g0, f_g1, f_g2, f_g3))
    
def plot_all(option, DG_xz, DG_yz, DG_wz, DG_xyw, result, X_targets_test, Y_targets_test, W_targets_test, Z_targets_test, n_viz):
    [imxz, imyz, imwz, imxzy, imyzw, imwzx, rmxzyz, rmyzwz, rmwzxz, rmxzyzx, rmyzy, rmyzwzy, rmxzx, rmwzxzw, rmwzw] = result.get_all()
    [imzx, imzy, imzw, rmzxz, rmzyz, rmzwz] = result.get_z_domain()
    # infer domain -> z
    plot(option, X_targets_test, n_viz, imxz, option.infer_xz,'$z_1$','$z_2$')
    plot(option, Y_targets_test, n_viz, imyz, option.infer_yz,'$z_1$','$z_2$')
    plot(option, W_targets_test, n_viz, imwz, option.infer_wz,'$z_1$','$z_2$')#
    # infer z -> domian
    plot(option, X_targets_test, n_viz, imxzy, option.infer_xzy,'$y_1$','$y_2$')
    plot(option, Y_targets_test, n_viz, imyzw, option.infer_yzw,'$w_1$','$w_2$')
    plot(option, W_targets_test, n_viz, imwzx, option.infer_wzx,'$x_1$','$x_2$') #   
    # infer from z z -> domian
    plot(option, Z_targets_test, n_viz, imzy, option.infer_zy,'$y_1$','$y_2$')
    plot(option, Z_targets_test, n_viz, imzw, option.infer_zw,'$w_1$','$w_2$')
    plot(option, Z_targets_test, n_viz, imzx, option.infer_zx,'$x_1$','$x_2$') #      
    # infer from z z -> domian   ->  other  -> back to z to see 
    plot(option, Z_targets_test, n_viz, rmzyz, option.infer_zyz,'$y_1$','$y_2$')
    plot(option, Z_targets_test, n_viz, rmzwz, option.infer_zwz,'$w_1$','$w_2$')
    plot(option, Z_targets_test, n_viz, rmzxz, option.infer_zxz,'$x_1$','$x_2$') #   
    # rec domain -> z ****wrong!!!!
    plot(option, X_targets_test, n_viz, rmxzyz, option.reconstruct_xzyz,'$z_1$','$z_2$')
    plot(option, Y_targets_test, n_viz, rmyzwz, option.reconstruct_yzwz,'$z_1$','$z_2$')
    plot(option, W_targets_test, n_viz, rmwzxz, option.reconstruct_wzxz,'$z_1$','$z_2$')#
    # rec domain -> z -> domain itself
    plot(option, X_targets_test, n_viz, rmxzx, option.reconstruct_xzx,'$x_1$','$x_2$')
    plot(option, Y_targets_test, n_viz, rmyzy, option.reconstruct_yzy,'$y_1$','$y_2$')
    plot(option, W_targets_test, n_viz, rmwzw, option.reconstruct_wzw,'$w_1$','$w_2$')#

    # rec domain -> z - > target domain -> z -> domain itself
    plot(option, X_targets_test, n_viz, rmxzyzx, option.reconstruct_xzyzx,'$x_1$','$x_2$')
    plot(option, Y_targets_test, n_viz, rmyzwzy, option.reconstruct_yzwzy,'$y_1$','$y_2$')
    plot(option, W_targets_test, n_viz, rmwzxzw, option.reconstruct_wzxzw,'$w_1$','$w_2$')#

    plot_lr(option, option.lr_xz, DG_xz.get_FD(), DG_xz.get_FG(), "Reconstruction x", "Reconstruction z")
    plot_lr(option, option.lr_yz, DG_yz.get_FD(), DG_yz.get_FG(), "Reconstruction y", "Reconstruction z")
    plot_lr(option, option.lr_wz, DG_wz.get_FD(), DG_wz.get_FG(), "Reconstruction w", "Reconstruction z")
    #plot_lr(option, option.lr_xyw, DG_xyw.get_FD(), DG_xyw.get_FG(), "Reconstruction x", "Reconstruction y")