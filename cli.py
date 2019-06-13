import configparser
import os
import argparse
from dnn_packages.ml_data_modeling_test.ml_models.stacked_autoencoder import stacked_autoencoder

class config_models(object):

    def __init__(self):
        pass

    def ConfigSectionMap(self, section, file):
        dict = {}
        Config = configparser.ConfigParser()
        Config.read(os.getcwd()+'/ml_data_modeling_test/config_files/'+file)
        options = Config.options(section)
        for option in options:
            try:
                dict[option] = Config.get(section, option)
                if dict[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict[option] = None

        return dict


    def main(self):

        parser = argparse.ArgumentParser(description= 'Autoencoder')

        parser.add_argument('--config_file', type=int, default=1)

        parser.add_argument('--embedding_dim', type=int, default=200)
        parser.add_argument('--margin_value', type=float, default=1.0)
        parser.add_argument('--score_func', type=str, default='L1')
        parser.add_argument('--batch_size', type=int, default=4800)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--n_generator', type=int, default=24)
        parser.add_argument('--n_rank_calculator', type=int, default=24)
        parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
        parser.add_argument('--summary_dir', type=str, default='../summary/')
        parser.add_argument('--max_epoch', type=int, default=500)
        parser.add_argument('--eval_freq', type=int, default=10)

        args = parser.parse_args()

        if args.config_file:
            batch = self.ConfigSectionMap("LSTM", "models_config.ini")['batch']



            stacked_autoencoder.create_autoencoder_model()
            print(Name)
        else:
            print(args)


if __name__ == '__main__':
    config = config_models()
    config.main()
