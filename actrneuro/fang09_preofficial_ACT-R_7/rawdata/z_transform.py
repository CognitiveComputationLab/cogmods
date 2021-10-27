from sklearn import preprocessing

def z_transformation(data):

    scaler = preprocessing.StandardScaler().fit(data.reshape(-1,1))
    standart = scaler.transform(data.reshape(-1,1))
    return standart.flatten()