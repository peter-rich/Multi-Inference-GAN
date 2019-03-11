import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--n_iter', type=int, default=30, help='numbers of iter you want to train ')
        self.parser.add_argument('--n_epoch', type=int, default=60, help='numbers of epoch you want to train in the initial training')
        self.parser.add_argument('--n_epoch_2', type=int, default=50, help='numbers of epoch you want to train after the initial training')
        self.parser.add_argument('--n_epoch_3', type=int, default=3, help='numbers of epoch you want to train after the initial training')
        self.parser.add_argument('--n_epoch_4', type=int, default=5, help='numbers of epoch you want to train after the initial training')
        self.parser.add_argument('--n_epoch_5', type=int, default=2, help='numbers of epoch you want to train after the initial training')
        self.parser.add_argument('--n_epoch_6', type=int, default=1, help='numbers of epoch you want to train after the initial training')
       
        self.parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        self.parser.add_argument('--dataset_size_x', type=int, default=512*4, help='Scale of the X datesets')
        self.parser.add_argument('--dataset_size_y', type=int, default=512*4, help='Scale of the Y datesets')
        self.parser.add_argument('--dataset_size_z', type=int, default=512*4, help='Scale of the Z datesets')
        self.parser.add_argument('--dataset_size_x_test', type=int, default=512*2, help='Size of the X test datesets')
        self.parser.add_argument('--dataset_size_y_test', type=int, default=512*2, help='Size of the Y test datesets')
        self.parser.add_argument('--dataset_size_z_test', type=int, default=512*2, help='Size of the Z test datesets')
        self.parser.add_argument('--input_dim', type=int, default=2, help='The size of input_dim')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='The size of latent_dim')
        self.parser.add_argument('--eps_dim', type=int, default=2, help='The size of eps_dim')

        self.parser.add_argument('--load_model', type=int, default=0, help='TO load the number of model')

        self.parser.add_argument('--n_layer_disc', type=int, default=2, help='The size of discriminator')
        self.parser.add_argument('--n_hidden_disc', type=int, default=256, help='The size of hidden_dim of discriminator network')
        self.parser.add_argument('--n_layer_gen', type=int, default=2, help='The number of generative layer')

        self.parser.add_argument('--n_hidden_gen', type=int, default=256, help='The size of hidden_dim of generative network')
        self.parser.add_argument('--n_layer_inf', type=int, default=2, help='The number of infering layer')
        self.parser.add_argument('--n_hidden_inf', type=int, default=256, help='The size of hidden_dim of infering network')

        self.parser.add_argument('--std_x', type=float, default=0.04, help='The size of hidden_dim of generative network')
        self.parser.add_argument('--std_y', type=float, default=0.04, help='The number of infering layer')
        self.parser.add_argument('--std_z', type=float, default=1, help='The size of hidden_dim of infering network')


        self.parser.add_argument('--gpu_id', type=str, default='0', help='The execution id of GPU')
        #self.parser.add_argument('--model_path', type=str, default='', help='The initial of trained x')
        self.parser.add_argument('--x_train', type=str, default='X_gmm_data_train.pdf', help='The initial of trained x')
        self.parser.add_argument('--y_train', type=str, default='Y_gmm_data_train.pdf', help='The initial of trained y')
        self.parser.add_argument('--z_train', type=str, default='Z_gmm_data_train.pdf', help='The initial of trained z')

        self.parser.add_argument('--infer_xz', type=str, default='inferred_xz.pdf', help='The initial of infer xz')
        self.parser.add_argument('--infer_yz', type=str, default='inferred_yz.pdf', help='The initial of infer yz')
        self.parser.add_argument('--infer_zx', type=str, default='inferred_zx.pdf', help='The initial of infer zx')
        self.parser.add_argument('--infer_zy', type=str, default='inferred_zy.pdf', help='The initial of infer zy')
        self.parser.add_argument('--infer_yzx', type=str, default='inferred_yzx.pdf', help='The initial of infer zx')
        self.parser.add_argument('--infer_xzy', type=str, default='inferred_xzy.pdf', help='The initial of infer zy')

        self.parser.add_argument('--reconstruct_yzxz', type=str, default='reconstruct_yzxz.pdf', help='The initial of Reconstruct xz')
        self.parser.add_argument('--reconstruct_xzyz', type=str, default='reconstruct_xzyz.pdf', help='The initial of Reconstruct yz')
        self.parser.add_argument('--reconstruct_xzyzx', type=str, default='reconstruct_xzyzx.pdf', help='The initial of infer yzx')
        self.parser.add_argument('--reconstruct_yzxzy', type=str, default='reconstruct_yzxzy.pdf', help='The initial of infer xzy')
        self.parser.add_argument('--reconstruct_xzx', type=str, default='reconstruct_xzx.pdf', help='The initial of infer xzx')
        self.parser.add_argument('--reconstruct_yzy', type=str, default='reconstruct_yzy.pdf', help='The initial of infer yzy')
        self.parser.add_argument('--reconstruct_zxz', type=str, default='reconstruct_zxz.pdf', help='The initial of infer xzx')
        self.parser.add_argument('--reconstruct_zyz', type=str, default='reconstruct_zyz.pdf', help='The initial of infer yzy')

        self.parser.add_argument('--x_test', type=str, default='X_gmm_data_test.pdf', help='The initial of test x')
        self.parser.add_argument('--y_test', type=str, default='Y_gmm_data_test.pdf', help='The initial of test y')
        self.parser.add_argument('--z_test', type=str, default='Z_gmm_data_test.pdf', help='The initial of test z')


        self.parser.add_argument('--lr_xz', type=str, default='learning_curves_xz.pdf', help='The initial of XZ Learning rate')
        self.parser.add_argument('--lr_yz', type=str, default='learning_curves_yz.pdf', help='The initial of YZ Learning rate')
        self.parser.add_argument('--lr_xy', type=str, default='learning_curves_xy.pdf', help='The initial of XY Learning rate')

        self.parser.add_argument('--result_dir', type=str, default='results/l2_x_z_y/', help='The dir of the result_dir')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        """ Create directory for results """
        
        directory = self.opt.result_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        return self.opt
