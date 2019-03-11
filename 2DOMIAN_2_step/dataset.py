from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

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
    mean_x = map(lambda x:  np.array(x), [[2, 0],
                                     [-2, 0],
                                     [0, 0],
                                     [0, 2],
                                     [0, -2]])

    mean_y = map(lambda y:  np.array(y),[[0, 0],
                                     [2, 2],
                                     [-2, -2],
                                     [2, -2],
                                     [-2, 2]
                                     ])

    mean_z = map(lambda z:  np.array(z),[[0, 0]])
    
    """ Create dataset """
    data_x = BaseData()
    data_y = BaseData()
    data_z = BaseData()
    data_x.init_dataset(opt, mean_x, opt.std_x, opt.x_train, opt.x_test)
    data_y.init_dataset(opt, mean_y, opt.std_y, opt.y_train, opt.y_test)
    data_z.init_dataset(opt, mean_z, opt.std_z, opt.z_train, opt.z_test)
    return data_x, data_y, data_z

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

def print_xy(epoch, i, f_d, f_g0, f_g1, f_g2, f_g3):
    print("epoch %d iter %d: discloss %f genloss %f adv_x %f recons_x %f recons_y %f" % (epoch, i, f_d, f_g0, f_g1, f_g2, f_g3))
    
def plot_all(option, DG_xz, DG_yz, DG_xy, im, rm, X_targets_test, Y_targets_test, Z_targets_test, n_viz):
    imxz=im[0]
    imzx=im[1]
    imyzx=im[2]
    imyz=im[3]
    imzy=im[4]
    imxzy=im[5]

    rmxzyzx=rm[0]
    rmyzy=rm[1]
    rmxzyz=rm[2]
    rmyzxzy=rm[3]
    rmxzx=rm[4]
    rmyzxz=rm[5]
    rmzxz=rm[6]
    rmzyz=rm[7]

    plot(option, X_targets_test, n_viz, imxz, option.infer_xz,'$z_1$','$z_2$')
    plot(option, X_targets_test, n_viz, rmxzx, option.reconstruct_xzx,'$x_1$','$x_2$')
    plot(option, Z_targets_test, n_viz, imzx, option.infer_zx,'$x_1$','$x_2$')
    plot(option, X_targets_test, n_viz, imxzy, option.infer_xzy,'$y_1$','$y_2$')
    plot(option, X_targets_test, n_viz, rmxzyzx, option.reconstruct_xzyzx,'$x_1$','$x_2$')
    plot(option, X_targets_test, n_viz, rmxzyz, option.reconstruct_xzyz,'$z_1$','$z_2$')

    plot(option, Y_targets_test, n_viz, imyz, option.infer_yz,'$z_1$','$z_2$')
    plot(option, Y_targets_test, n_viz, rmyzy, option.reconstruct_yzy,'$y_1$','$y_2$')
    plot(option, Z_targets_test, n_viz, imzy, option.infer_zy,'$y_1$','$y_2$')
    plot(option, Y_targets_test, n_viz, imyzx, option.infer_yzx,'$x_1$','$x_2$')
    plot(option, Y_targets_test, n_viz, rmyzxzy, option.reconstruct_yzxzy,'$y_1$','$y_2$')
    plot(option, Y_targets_test, n_viz, rmyzxz, option.reconstruct_yzxz,'$z_1$','$z_2$')
    
    plot(option, Z_targets_test, n_viz, rmzxz, option.reconstruct_zxz ,'$z_1$','$z_2$')
    plot(option, Z_targets_test, n_viz, rmzyz, option.reconstruct_zyz ,'$z_1$','$z_2$')
    
    #plot_lr(option, option.lr_xz, DG_xz.get_FD(), DG_xz.get_FG(), "Reconstruction x", "Reconstruction z")
    #plot_lr(option, option.lr_yz, DG_yz.get_FD(), DG_yz.get_FG(), "Reconstruction y", "Reconstruction z")
    #plot_lr(option, option.lr_xy, DG_xy.get_FD(), DG_xy.get_FG(), "Reconstruction x", "Reconstruction y")