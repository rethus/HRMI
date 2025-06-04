import warnings
import time

from numpy import linalg as LA
from sklearn.cluster import KMeans

from utilities import *

from utils.dataloader import WSDREAM_1_MatrixDataset, ToTorchDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, rmse
from utils.loss import MILoss
from models.HRMI.model import HRMIModel


# freeze_random()  # 冻结随机数 保证结果一致


def execute(matrix, I_outlier, user_country_matrix, service_country_matrix,
            user_country_one_hot, service_country_one_hot, density, para):
    start_time = time.time()
    type_ = para['dataType']
    num_user = matrix.shape[0]
    num_service = matrix.shape[1]
    rounds = para['rounds']
    # rounds = 10
    ku = para['ku']
    ks = para['ks']
    lr = para['lr']
    epochs = para['epochs']
    dim = para['dim']
    num_encoder_layer = para['num_encoder_layer']
    g = para['g']
    loss_l = para['loss_l']
    loss_a = para['loss_a']

    logger.info('Data matrix size: %d users * %d services' % (num_user, num_service))
    logger.info('Run the algorithm for %d rounds: matrix density = %.3f.' % (rounds, density))
    eval_results = np.zeros((rounds, len(para['metrics'])))
    time_results = np.zeros((rounds, 1))
    maes = []
    rmses = []
    for k in range(rounds):
        iter_start_time = time.time()
        logger.info('----------------------------------------------')
        logger.info('%d-round starts.' % (k + 1))
        logger.info('----------------------------------------------')

        rt_data = WSDREAM_1_MatrixDataset(type_)
        train_data, test_data = rt_data.split_train_test(density)

        train_dataset = ToTorchDataset(train_data)
        test_dataset = ToTorchDataset(test_data)
        train_dataloader = DataLoader(train_dataset, batch_size=64, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)

        (train_matrix, test_matrix) = getMatrix(matrix, train_data, test_data)
        logger.info('Removing data entries done.')

        _, user_reputation_list = get_user_reputation_weight_matrix(para['theta'], user_country_matrix, train_matrix, ku)
        _, service_reputation_list = get_service_reputation_weight_matrix(para['theta'], service_country_matrix, train_matrix, ks)

        loss_fn = MILoss(loss_l, loss_a)

        HRMI = HRMIModel(loss_fn, rt_data.row_n, rt_data.col_n, dim, user_reputation_list, service_reputation_list,
                         user_country_one_hot, service_country_one_hot, g, num_encoder_layer=num_encoder_layer)

        opt = Adam(HRMI.parameters(), lr=lr, weight_decay=1e-5)

        HRMI.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader, save_filename=f"Density_{density}")
        y, y_pred = HRMI.predict(test_dataloader, True)

        mae_ = mae(y, y_pred)
        rmse_ = rmse(y, y_pred)

        maes.append(mae_)
        rmses.append(rmse_)

        HRMI.logger.info(
            f"Density:{density:.3f}, type:{type_}, mae:{mae_:.4f}, rmse:{rmse_:.4f}")

        eval_results[k, :] = [mae_, rmse_]
        time_results[k] = time.time() - iter_start_time

        logger.info(f'{k + 1}-round done. Running time: {time_results[k]} sec')
        logger.info('----------------------------------------------')

    mae_tot, rmse_tot = 0, 0
    for i in range(len(maes)):
        mae_tot += maes[i]
        rmse_tot += rmses[i]

    logger.info(f'mae: {mae_tot / len(maes): .4f}, rmse: {rmse_tot / len(maes): .4f}')

    out_file = '%s%sResult_%.2f.txt' % (para['outPath'], para['dataType'], density)

    saveResult(out_file, eval_results, time_results, para)
    logger.info('Config density = %.3f done. Running time: %.2f sec'
                % (density, time.time() - start_time))
    logger.info('==============================================')


def get_user_reputation_weight_matrix(theta, user_country_matrix, train_matrix, ku):
    num_service = train_matrix.shape[1]
    num_user = train_matrix.shape[0]
    reputation = np.zeros((num_user, 2), dtype=int)
    for sid in range(0, num_service):
        try:
            data = train_matrix[:, sid]
            real_idx = np.where(data > 0)
            real_data = data[real_idx].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(ku).fit(real_data)
            labels = km.labels_
            _label = np.argmax(np.bincount(labels))
            max_cluster = real_data[labels == _label]
            mean = max_cluster.mean()
            std = max_cluster.std()
            for uid in real_idx[0]:
                if abs(train_matrix[uid][sid] - mean) <= 3 * std:
                    reputation[uid][0] += 1
                else:
                    reputation[uid][1] += 1
        except Exception as e:
            pass

    user_reputation = (np.exp(theta * reputation[:, 0])) / (np.exp(theta * reputation[:, 0]) + np.exp(theta * reputation[:, 1]))
    weight_matrix = user_country_matrix * user_reputation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight_matrix = weight_matrix / weight_matrix.sum(1).reshape(1, -1)
    weight_matrix[np.isnan(weight_matrix)] = 0
    return weight_matrix, user_reputation


def get_service_reputation_weight_matrix(theta, service_country_matrix, train_matrix, ks):
    num_service = train_matrix.shape[1]
    num_user = train_matrix.shape[0]
    reputation = np.zeros((num_service, 2), dtype=int)
    for uid in range(0, num_user):
        try:
            data = train_matrix[uid, :]
            real_idx = np.where(data > 0)
            real_data = data[real_idx].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(ks).fit(real_data)
            labels = km.labels_
            _label = np.argmax(np.bincount(labels))
            # maximum cluster
            max_cluster = real_data[labels == _label]
            mean = max_cluster.mean()
            std = max_cluster.std()
            for sid in real_idx[0]:
                if abs(train_matrix[uid][sid] - mean) <= 3 * std:
                    reputation[sid][0] += 1
                else:
                    reputation[sid][1] += 1
        except Exception as e:
            pass

    service_reputation = (np.exp(theta * reputation[:, 0])) / (np.exp(theta * reputation[:, 0]) + np.exp(theta * reputation[:, 1]))
    weight_matrix = service_country_matrix * service_reputation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight_matrix = weight_matrix / weight_matrix.sum(1).reshape(1, -1)
    weight_matrix[np.isnan(weight_matrix)] = 0
    return weight_matrix, service_reputation


def merge_test_outlier(I_outlier, test_matrix):
    merge_matrix = test_matrix.copy()
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            if test_matrix[i][j] != 0 and I_outlier[i][j] == 1:
                merge_matrix[i][j] = 0
    return merge_matrix


def getMatrix(matrix, train_data, test_data):
    trainMatrix = np.zeros(matrix.shape)
    for row in train_data:
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        trainMatrix[uid][iid] = rate

    testMatrix = np.zeros(matrix.shape)
    for row in test_data:
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        testMatrix[uid][iid] = rate

    return trainMatrix, testMatrix


def errMetric(testMatrix, predMatrix, metrics):
    result = []
    (testVecX, testVecY) = np.where(testMatrix)
    testVec = testMatrix[testVecX, testVecY]
    predVec = predMatrix[testVecX, testVecY]
    absError = np.absolute(predVec - testVec)
    mae = np.average(absError)
    for metric in metrics:
        if 'MAE' == metric:
            result = np.append(result, mae)
        elif 'RMSE' == metric:
            rmse = LA.norm(absError) / np.sqrt(absError.size)
            result = np.append(result, rmse)
    return result
