import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.optimize import minimize


def qods(x=0.15, y=0.16, r=0.5, eps=0.1):
    # x = 0.15
    # y = 0.26
    # r = 0.5
    not_x = 1 - x
    not_y = 1 - y
    not_r = 1 - r
    # eps = 0.17

    """
    Turn q card 
    """
    p_q_r = x * (1 - eps) * y * r
    p_q_not_r = x * not_r
    p_not_q_r = x * eps * not_y * r
    p_not_q_not_r = x * not_r
    not_p_q_r = 1 - x * (1 - eps) * y * r
    not_p_q_not_r = not_x * not_r
    not_p_not_q_r = 1 - x * eps * not_y * r
    not_p_not_q_not_r = not_x * not_r

    """
    Turn p card 
    """
    q_p_r = (1 - eps) * x * r
    q_p_not_r = y * not_r
    q_not_p_r = y * not_x * r
    q_not_p_not_r = y * not_r
    not_q_p_r = eps * x * r
    not_q_p_not_r = not_y * not_r
    not_q_not_p_r = not_y * not_x * r
    not_q_not_p_not_r = not_y * not_r

    """
    Turn p and not p card 
    Probabilities for both hypothesis
    """
    p_q = p_q_r + p_q_not_r
    not_p_q = not_p_q_r + not_p_q_not_r
    p_not_q = p_not_q_r + p_not_q_not_r
    not_p_not_q = not_p_not_q_r + not_p_not_q_not_r

    """
    Turn q and not q card 
    Probabilities for both hypothesis
    """
    q_p = q_p_r + q_p_not_r
    not_q_p = not_q_p_r + not_q_p_not_r
    q_not_p = q_not_p_r + q_not_p_not_r
    not_q_not_p = not_q_not_p_r + not_q_not_p_not_r

    p_r = x * r
    p_not_r = x * not_r
    q_r = y * r
    q_not_r = y * not_r
    not_p_r = not_x * r
    not_p_not_r = not_x * not_r
    not_q_r = not_y * r
    not_q_not_r = not_y * not_r

    # information P
    a = p_q_r * np.log2((p_q_r * x) / (p_q * p_r))
    b = p_q_not_r * np.log2((p_q_not_r * x) / (p_q * p_not_r))
    c = p_not_q_r * np.log2((p_not_q_r * x) / (p_not_q * p_r))
    d = p_not_q_not_r * np.log2((p_not_q_not_r * x) / (p_not_q * p_not_r))
    information_p = a + b + c + d

    a = not_p_q_r * np.log2((not_p_q_r * not_x) / (not_p_q * not_p_r))
    b = not_p_q_not_r * np.log2((not_p_q_not_r * not_x) / (not_p_q * not_p_not_r))
    c = not_p_not_q_r * np.log2((not_p_not_q_r * not_x) / (not_p_not_q * not_p_r))
    d = not_p_not_q_not_r * np.log2((not_p_not_q_not_r * not_x) / (not_p_not_q * not_p_not_r))
    information_not_p = a + b + c + d

    a = q_p_r * np.log2((q_p_r * y) / (q_p * q_r))
    b = q_p_not_r * np.log2((q_p_not_r * y) / (q_p * q_not_r))
    c = q_not_p_r * np.log2((q_not_p_r * y) / (q_not_p * q_r))
    d = q_not_p_not_r * np.log2((q_not_p_not_r * y) / (q_not_p * q_not_r))
    information_q = a + b + c + d

    a = not_q_p_r * np.log2((not_q_p_r * not_y) / (not_q_p * not_q_r))
    b = not_q_p_not_r * np.log2((not_q_p_not_r * not_y) / (not_q_p * not_q_not_r))
    c = not_q_not_p_r * np.log2((not_q_not_p_r * not_y) / (not_q_not_p * not_q_r))
    d = not_q_not_p_not_r * np.log2((not_q_not_p_not_r * not_y) / (not_q_not_p * not_q_not_r))
    information_not_q = a + b + c + d

    sum_all = [information_p, information_q, information_not_p, information_not_q]

    scaled_inf_p = information_p / np.sum(sum_all)
    scaled_inf_q = information_q / np.sum(sum_all)
    scaled_inf_not_p = information_not_p / np.sum(sum_all)
    scaled_inf_not_q = information_not_q / np.sum(sum_all)

    return scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q


def stf(card):
    x = -2.37 + 9.06 * card
    return 1 / (1 + np.exp(x))


"""
optimize RMSE of QODS pred
"""
def optimize_inf_model(params, *args):
    x, y, r, eps = params
    obs_p, obs_not_p, obs_q, obs_not_q = args
    scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(x, y, r, eps)
    error = (obs_p - stf(scaled_inf_p)) ** 2 + (obs_not_p - stf(scaled_inf_not_p)) ** 2 + (
            obs_q - stf(scaled_inf_q)) ** 2 + (obs_not_q - stf(
        scaled_inf_not_q)) ** 2
    return np.sqrt(error) / 4


def testP():
    for i in range(10, 100000000, 100):
        tmp = 1 / (i)
        scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(tmp, .20, .50, .1)
        print("P(p): ", tmp)
        print("prob p:", stf(scaled_inf_p))
        print("prob not p:", stf(scaled_inf_not_p))
        print("prob q:", stf(scaled_inf_q))
        print("prob not q:", stf(scaled_inf_not_q))


def gen_data():
    data_p = [[[] for j in range(100)] for i in range(100)]
    data_not_p = [[[] for j in range(100)] for i in range(100)]
    data_q = [[[] for j in range(100)] for i in range(100)]
    data_not_q = [[[] for j in range(100)] for i in range(100)]
    for i in range(1, 100):
        for j in range(1, 100):
            scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(i * 0.01, j * 0.01, .5, .1)
            data_p[i - 1][j - 1] = scaled_inf_p
            data_not_p[i - 1][j - 1] = scaled_inf_not_p
            data_q[i - 1][j - 1] = scaled_inf_q
            data_not_q[i - 1][j - 1] = scaled_inf_not_q
    df_p = pd.DataFrame(data_p)
    df_not_p = pd.DataFrame(data_not_p)
    df_q = pd.DataFrame(data_q)
    df_not_q = pd.DataFrame(data_not_q)
    df_p.to_csv('odsP.csv')  # , index=False)
    df_not_p.to_csv('odsNotP.csv')  # , index=False)
    df_q.to_csv('odsQ.csv')  # , index=False)
    df_not_q.to_csv('odsNotQ.csv')  # , index=False)


def gen_RAST_data():
    data = []
    for i in range(1, 99):
        scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(0.01, 0.9, .5, .01)  # (i*0.01)-0.001
        data.append((i, stf(scaled_inf_q), stf(scaled_inf_not_q)))
    df = pd.DataFrame(data)
    return df


def test_gen_data():
    data = []
    for i in range(1, 100):
        for j in range(1, 100):
            scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(i * 0.01, j * 0.01, .5, .1)
            data.append((i, j, stf(scaled_inf_not_q)))
    df = pd.DataFrame(data)
    # df.to_csv('test.csv', index=False)
    return df
    #


def generate_colormap_Plot(df):
    cmap = cm.get_cmap('Greys')
    fig, ax = plt.subplots(1)
    # Now here's the plot. range(len(df)) just makes the x values 1, 2, 3...
    # df[0] is then the y values. c sets the colours (same as y values in this
    # case). s is the marker size.
    ax.scatter(df[0], df[1], c=(df[2] * 100), s=120, cmap=cmap, edgecolor='None')
    plt.show()


def generate_RAST_Plot(df):
    plt.plot(df[0], df[1], 'r', label='q')
    plt.plot(df[0], df[2], 'g', label='not q')
    plt.xlabel("P(q)")
    plt.ylabel("Probability of selecting a card")
    plt.legend(loc='right')
    plt.show()


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def plotdata(df):
    contour = plt.tricontourf(df[0], df[1], df[2], 100, cmap="Greys")
    cbar = plt.colorbar(contour, format=ticker.FuncFormatter(fmt))
    cbar.set_label('')

    plt.xlabel("P(p)")
    plt.ylabel("P(q)")
    plt.show()


"""
Optimize Data from Excelfile and save it in csv
"""


def optimizeQODS(data_file_source, data_file_output):
    results = []
    initial_values = np.array([0.01, 0.02, 0.5, 0.1])
    prob_bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    df = pd.read_csv(data_file_source, header=None, sep=";")
    df = df.apply(lambda x: x.str.replace(',', '.'))
    df = df.apply(pd.to_numeric)
    for index, row in df.iterrows():
        data = (row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3])
        tmp = minimize(optimize_inf_model, x0=initial_values, args=data, method='SLSQP', bounds=prob_bounds,
                       constraints=({'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.001}))
        results.append(tmp.x)
    df1 = pd.DataFrame(results)
    df1 = df1.round(5)
    df1.to_csv(data_file_output, index=None)
    return df1


"""
Calc predition from data
"""


def calcPred(data_file_source, data_file_output):
    df = pd.read_csv(data_file_source, header=None, sep=",")
    pred = []
    for index, row in df.iterrows():
        tmp = []
        if row.iloc[0] == 1:
            row.iloc[0] = row.iloc[0] - 0.0000001
        elif row.iloc[0] == 0:
            row.iloc[0] = row.iloc[0] + 0.0000001
        if row.iloc[1] == 1:
            row.iloc[1] = row.iloc[1] - 0.0000001
        elif row.iloc[1] == 0:
            row.iloc[1] = row.iloc[1] + 0.0000001
        if row.iloc[2] == 1:
            row.iloc[2] = row.iloc[2] - 0.0000001
        elif row.iloc[2] == 0:
            row.iloc[2] = row.iloc[2] + 0.0000001
        if row.iloc[3] == 1:
            row.iloc[3] = row.iloc[3] - 0.0000001
        elif row.iloc[3] == 0:
            row.iloc[3] = row.iloc[3] + 0.0000001
        scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(row.iloc[0], row.iloc[1], row.iloc[2],
                                                                              row.iloc[3])
        tmp.append(stf(scaled_inf_p))
        tmp.append(stf(scaled_inf_not_p))
        tmp.append(stf(scaled_inf_q))
        tmp.append(stf(scaled_inf_not_q))
        pred.append(tmp)
    df2 = pd.DataFrame(pred)
    df2.to_csv(data_file_output, header=None)
    return pred


"""
quick test of params
"""


def showValues(p, q, r, eps):
    scaled_inf_p, scaled_inf_not_p, scaled_inf_q, scaled_inf_not_q = qods(p, q, r, eps)
    print(stf(scaled_inf_p))
    print(stf(scaled_inf_not_p))
    print(stf(scaled_inf_q))
    print(stf(scaled_inf_not_q))





# test123 = minimize(optimize_inf_model, x0=initial_values, args=data, method='SLSQP' ,bounds=prob_bounds, constraints=({'type': 'ineq', 'fun': lambda x:  x[1]-x[0] - 0.001}))
# a, b, c = fmin_l_bfgs_b(optimize_inf_model, x0=initial_values, args=data, bounds=prob_bounds, approx_grad=True)


optimizeQODS('/Users/matze/Nextcloud/Bachelorthesis/Bachelorarbeit_daten/qods_data/qods_neg_obs.csv', '/Users/matze/Nextcloud/Bachelorthesis/Bachelorarbeit_daten/qods_data/qods_neg_opt_params.csv')
calcPred('/Users/matze/Nextcloud/Bachelorthesis/Bachelorarbeit_daten/qods_data/qods_neg_opt_params.csv', '/Users/matze/Nextcloud/Bachelorthesis/Bachelorarbeit_daten/qods_data/qods_neg_pred.csv')
