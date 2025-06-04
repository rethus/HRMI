import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import IsolationForest


def get_outlier_matrix(dataType, outliers_fraction):
    '''
    output: Indicator Matrix (0: normal; 1: outlier)
    '''
    R = load_data(f"data/wsdream_1/{dataType}Matrix.txt")
    m, n = R.shape
    I_outlier = np.zeros([m, n])
    rng = np.random.RandomState(42)
    x = []
    x_ind = []
    for i in range(m):
        for j in range(n):
            if R[i][j] >= 0:
                x.append(R[i][j])
                x_ind.append(i * n + j)

    x = np.array(x)
    x = x.reshape(-1, 1)

    clf = IsolationForest(max_samples=len(x), random_state=rng, contamination=outliers_fraction)
    clf.fit(x)
    y_pred_train = clf.predict(x)
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            row = int(x_ind[i] / n)
            col = int(x_ind[i] % n)
            I_outlier[row][col] = 1

    filename = f"data/wsdream_1/Full_I_outlier_{dataType}.npy"
    np.save(filename, I_outlier)
    print(f"============outliers_fraction {dataType} %s DONE ==============" % outliers_fraction)


def load_data(filedir):
    R = []
    with open(filedir) as fin:
        for line in fin:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R


def get_user_country_matrix():
    userlist = pd.read_csv("data/wsdream_1/userlist.txt", sep="\t")
    num_user = userlist.shape[0]

    matrix = np.zeros((num_user, num_user), dtype=int)
    country_user = userlist.groupby('[Country]')['[User ID]'].unique()
    for key in country_user.keys():
        arr = country_user[key]
        pairs = list(itertools.permutations(arr, 2))
        for u, v in pairs:
            matrix[u][v] = 1
    np.save("data/wsdream_1/user_country_matrix.npy", matrix)


def get_service_country_matrix():
    userlist = pd.read_csv("data/wsdream_1/wslist.txt", sep="\t")
    num_user = userlist.shape[0]

    matrix = np.zeros((num_user, num_user), dtype=int)
    country_user = userlist.groupby('[Country]')['[Service ID]'].unique()
    for key in country_user.keys():
        arr = country_user[key]
        pairs = list(itertools.permutations(arr, 2))
        for u, v in pairs:
            matrix[u][v] = 1
    np.save("data/wsdream_1/service_country_matrix.npy", matrix)


def get_one_hot_vector(type):
    if type != 'user': type = 'ws'
    userlist = pd.read_csv(f"data/wsdream_1/{type}list.txt", sep="\t")
    num_user = userlist.shape[0]
    user_country_categories = []
    mp = {}
    for i in userlist['[Country]']:
        if i in mp:
            continue
        else:
            mp[i] = 1
            user_country_categories.append(i)
    print(user_country_categories)
    user_country_category_to_int = {}
    for i in range(len(user_country_categories)):
        user_country_category_to_int[user_country_categories[i]] = i
    print(user_country_category_to_int)
    user_country_int_encoded = [user_country_category_to_int[cat] for cat in user_country_categories]
    print(user_country_int_encoded)
    user_country_num_categories = len(user_country_category_to_int)
    # 创建一个零矩阵来存储独热编码结果
    user_country_one_hot_encoded = np.zeros((len(user_country_categories), user_country_num_categories))
    # 使用整数编码填充独热编码矩阵
    for i, code in enumerate(user_country_int_encoded):
        user_country_one_hot_encoded[i, code] = 1
    print(user_country_one_hot_encoded)
    # 将独热编码结果保存到文件
    user_country_one_hot_encoded_everyone = np.zeros((len(userlist), user_country_num_categories))
    for i in range(len(userlist)):
        if type == 'user':
            idx = userlist['[User ID]'][i]
        else:
            idx = userlist['[Service ID]'][i]
        coun = userlist['[Country]'][i]
        user_country_one_hot_encoded_everyone[idx] = user_country_one_hot_encoded[user_country_category_to_int[coun]]

    np.savetxt(f'data/wsdream_1/{type}_country_one_hot_encoded.txt', user_country_one_hot_encoded, fmt='%d')


if __name__ == '__main__':
    outliers_fraction = 0.1
    get_outlier_matrix("rt", outliers_fraction)
    get_user_country_matrix()
    get_service_country_matrix()
    get_one_hot_vector('rt')
    get_one_hot_vector('tp')
