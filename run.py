import sys
import time
import numpy as np

sys.path.append('src')

import src.evaluator as evaluator
import src.dataloader as dataloader
from src.utilities import logger, initConfig

para = {'dataType': 'rt',
        'dataPath': 'data/wsdream_1/',
        'outPath': 'results/',
        'metrics': ['MAE', 'RMSE'],
        'density': list(np.array([0.02])),
        'rounds': 1,  # how many runs are performed at each matrix density
        'dimension': 30,  # dimension of the latent factors
        'lambda': 5,  # regularization parameter
        'theta': 0.0001,
        'ku': 5,
        'ks': 15,
        'saveTimeInfo': False,  # whether to keep track of the running time
        'saveLog': False,  # whether to save log into file
        'debugMode': False,  # whether to record the debug info
        'parallelMode': False,
        'lr': 0.001,
        'epochs': 50,
        'dim': 16,
        'num_encoder_layer': 1,
        "g": 8,
        "loss_l": 0.05,
        "loss_a": 0.4
        }

def main():

    startTime = time.time()
    logger.info('==============================================')
    logger.info('Hybrid Reputation Fusion and Mutual Information Maximization for Web Services QoS Prediction.')
    # load the dataset
    initConfig(para)
    dataMatrix = dataloader.load(para)
    user_country_matrix = np.load(para['dataPath'] + 'user_country_matrix.npy')
    service_country_matrix = np.load(para['dataPath'] + 'service_country_matrix.npy')
    I_outlier = np.load(para['dataPath'] + 'Full_I_outlier_' + para['dataType'] + '.npy')
    user_country_one_hot = np.loadtxt('data/wsdream_1/user_country_one_hot_encoded_everyone.txt')
    service_country_one_hot = np.loadtxt('data/wsdream_1/ws_country_one_hot_encoded_everyone.txt')


    logger.info('Loading data done.')
    for density in para['density']:
        evaluator.execute(dataMatrix, I_outlier, user_country_matrix, service_country_matrix,
                          user_country_one_hot, service_country_one_hot,
                          density, para)
    logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
                              time.gmtime(time.time() - startTime)))
    logger.info('==============================================')
    sys.path.remove('src')


if __name__ == '__main__':
    main()
