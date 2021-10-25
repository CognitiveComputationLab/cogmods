import numpy as np
import sys

import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from rawdata.get_raw import get_raw
from rawdata.z_transform import z_transformation
from get_answers import get_answers, plot_answers
from copy import *


def evaluate(output_path="./plots", bold_step=1, plot=True):
    trails_lengths, trails_avg_times, trails_header, crashed_runs, runs, accuracy, meta = get_answers(model="fang6")


    model, ans, rt, lf, bold_scale, neg_bold_scale, bold_exp, neg_bold_exp, bold_positive, bold_negative, runs = meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], meta[6], meta[7], meta[8], meta[9], (int)(meta[10])

    if((runs - len(crashed_runs)) > 0):
        plot_answers(trails_lengths, trails_avg_times, trails_header, crashed_runs)


    # BOLD
    bold_table = np.zeros([13, 60])
    bold_table_divider = np.zeros([13, 60])
    boldfiles = [x[0] for x in os.walk("./log/bolds")]

    inslen = 0
    arrays = []

    limit = 14

    counter = 0
    for boldfile in sorted(boldfiles):

        if boldfile.count("/") < 5:
            continue

        if "training" in boldfile.lower():
            continue

        if int(boldfile.split("-")[1].split("/")[0]) in  crashed_runs:
            # print("this run is crashed and wont be checked", boldfile)
            continue

        counter = counter + 1
        if counter > limit:
            continue

        bold_result, buffer_list = get_bold(os.path.abspath(boldfile + "/bold-response.dat"))
        inslen = inslen + len(bold_result)



        permutation = np.argsort(buffer_list)


        for bs_ix in range(len(bold_result)):
            bold_result[bs_ix] = bold_result[bs_ix, permutation]
        buffer_list = buffer_list[permutation]




        arrays.append(bold_result)

    # plot_means(buffer_list, arrays, output_path, [0, 2, 6, 10], bold_step, 4)


    print("****************************************************")
    print("****************************************************")
    print("*************** RESULT OVERVIEW ********************")
    print("****************************************************")
    print("avg running time: " + shorten(str(trails_avg_times[0][0])) + " with std: " + shorten(str(trails_avg_times[0][1])))
    print("accuracy:           " + accuracy + " %")

    print("run amount:         " + str(runs))
    print("crashed run amount: " + str(len(crashed_runs)))
    print("----> crashed runs: " + str(len(crashed_runs)/runs*100) + " %")
    print("****************************************************")
    print("****************************************************")
    print("****************************************************")



    if int(len(crashed_runs)/runs*100) == 100:
        print("al crashed")
        return 0

    return calc_correlation(buffer_list, arrays, output_path, runs=runs, model=model, ans=ans, rt=rt, lf=lf, bold_scale=bold_scale, neg_bold_scale=neg_bold_scale, bold_exp=bold_exp, neg_bold_exp=neg_bold_exp, bold_positive=bold_positive, bold_negative=bold_negative, plot=plot)


def calc_correlation(buffer_list, arrays, output_path, z_transform=True, runs=None, model=None, ans=None, rt=None, lf=None, bold_scale=None, neg_bold_scale=None, bold_exp=None, neg_bold_exp=None, bold_positive=None, bold_negative=None, plot=False):
    print("in plot grpah")

    # human data of fang experiment

    # check
    # ppc_right = np.array([849.80762987, 852.279069767, 850.075396825, 854.919435216, 850.769163763, 853.199088146, 853.135551948, 862.827242525, 853.344155844, 858.243355482, 852.103174603, 860.460963455, 855.746515679, 857.243161094, 857.047077922, 866.171096346, 856.516233766, 863.806478405, 860.619047619, 860.34551495, 858.995644599, 863.441489362, 854.982954545, 867.365448505])
    # retrieval_right = np.array([755.215909091, 758.90245478, 754.365432099, 753.233204134, 756.106368564, 755.91607565, 756.503156566, 759.322997416, 757.686237374, 759.356589147, 755.75617284, 753.367571059, 756.81097561, 756.741134752, 759.059974747, 762.742248062, 759.716540404, 760.647286822, 757.966666667, 756.210594315, 758.18902439, 758.959810875, 758.961489899, 764.401162791])
    # acc_right = np.array([758.423768939, 763.732073643, 757.097685185, 758.011627907, 766.905487805, 762.918882979, 760.055871212, 769.322674419, 759.036931818, 764.999031008, 759.069444444, 760.375, 767.104674797, 762.277925532, 760.111742424, 767.899224806, 758.411458333, 765.466569767, 758.225925926, 759.310562016, 765.847560976, 761.857712766, 759.099431818, 768.20251938])
    # apfc_right = np.array([634.025, 637.593023256, 632.58, 634.806976744, 633.963414634, 633.282978723, 631.606818182, 640.320930233, 635.722727273, 638.765116279, 633.033333333, 633.593023256, 633.058536585, 633.485106383, 630.652272727, 640.637209302, 633.643181818, 638.830232558, 633.851111111, 635.623255814, 631.948780488, 633.008510638, 629.834090909, 639.376744186])
    # caudate_right = np.array([677.237373737, 680.855297158, 675.516049383, 679.183462532, 680.571815718, 676.279747833, 675.124579125, 686.311800172, 677.694444444, 681.427217916, 677.672427984, 678.825150732, 683.702800361, 678.613081166, 677.283670034, 684.131782946, 678.448653199, 681.610680448, 678.030452675, 680.424633936, 682.490514905, 679.486209614, 679.471380471, 683.621877692])
    # fusiform_right = np.array([787.806818182, 793.068313953, 786.366666667, 792.274709302, 789.579268293, 790.64162234, 786.116477273, 794.36627907, 787.893465909, 790.986918605, 787.961805556, 790.718023256, 790.448170732, 790.114361702, 786.691051136, 793.898255814, 787.310369318, 790.572674419, 785.926388889, 791.209302326, 790.537347561, 789.166888298, 784.277698864, 791.183139535])
    # motor_left = np.array([735.003787879, 740.361918605, 734.552777778, 736.039244186, 739.753556911, 735.750443262, 734.864109848, 743.519379845, 734.769886364, 740.001453488, 735.905555556, 734.908914729, 739.788109756, 736.608156028, 736.032670455, 743.364825581, 735.232481061, 740.071705426, 734.739351852, 735.771317829, 739.988821138, 736.906471631, 735.500473485, 746.035368217, 738.248106061, 744.066375969, 739.027314815, 738.549418605, 742.670223577, 739.154698582])

    # retrieval_right = z_transformation((retrieval_right[0::2] + retrieval_right[1::2]) / 2)
    # motor_left = z_transformation((motor_left[0::2] + motor_left[1::2]) / 2)

    # motor_left_error = [3] * len(motor_left)
    # retrieval_right_error = [3] * len(retrieval_right)


    # acc_right_mean, acc_right_error = get_raw("./rawdata/fang_acc_R_between.csv", 2, z_transform)



    # caudate_left_mean, caudate_right_error = get_raw("./rawdata/fang_caudate_L_between.csv", 2, z_transform)
    # oct_left_mean, caudateoct_left_error = get_raw("./rawdata/fang_otc_L_within.csv", 2, z_transform)
    # oct_right_mean, caudateoct_right_error = get_raw("./rawdata/fang_otc_R_within.csv", 2, z_transform)
    # pfc_right_mean, pfc_right_error = get_raw("./rawdata/fang_pfc_R_between.csv", 2, z_transform)
    # pfc_left_mean, pfc_left_error = get_raw("./rawdata/fang_pfc_L_between.csv", 2, z_transform)



    # retrieval_right = retrieval_right






    if plot:
        matplotlib.use("agg", force=True)
        plt.clf()
        plt.tight_layout()



    correlation_sum = 0
    correlation_counter = 0

    for b in range(len(buffer_list)):
        arrr = []

        buffer = buffer_list[b]



        for a in arrays:
            nparray = np.array(a)

            arrr.append(nparray[:, b])




        buffer = buffer.lower()



        pred_means, pred_stds, pred_lenghts = tolerant_mean(np.array(arrr), False)
        pred_means_z, pred_stds_z, pred_lenghts = tolerant_mean(np.array(arrr), True)

        print("***************** ++++ *****************************")
        print(buffer)
        print(pred_means_z, pred_stds_z, pred_lenghts)

        marker = "o"

        buffer = buffer.lower()



        if z_transform and buffer in ["retrieval", "production", "manual", "retrieval", "goal", "imaginal", "visual"]:

            if buffer == "visual":

                # visual_actr_right_mean, visual_actr_right_error = get_raw("./rawdata/actr_visual_R_NA.csv", 2, z_transform)
                # visual_actr_left_mean, visual_actr_left_error = get_raw("./rawdata/actr_visual_L_NA.csv", 2, z_transform)

                visual_actr_right_mean = np.array([ 1.11603323, -0.32731044,  0.78994743,  0.84912295, -0.11606774, -0.25845359, 0.94409792, 0.9232647, -0.74110511, -1.26020965, 0.36303498, -2.28235468])
                visual_actr_right_error = np.array([2.54767522, 2.50230802, 2.52718348, 2.51934423, 2.53369455, 2.50503333, 2.55174952, 2.54447777, 2.5429782, 2.50457096, 2.53003117, 2.50777911])
                visual_actr_left_mean = np.array([ 1.5883178,   0.09200375, -1.01275734, -0.22832655, 1.47756774, -0.49093257, -1.06741823,  0.11216549,  1.67004962, -0.26685687, -1.25228068, -0.62153217] )
                visual_actr_left_error = np.array([0.95549614, 0.94331072, 0.9395103, 0.94219211, 0.95156178, 0.94055036, 0.95110327, 0.9518787, 0.96216756, 0.93895223, 0.94213333, 0.93750646])

                container = []
                container.append([visual_actr_right_mean, visual_actr_right_error, "VISUAL Right ACT-R"])
                container.append([visual_actr_left_mean, visual_actr_left_error, "VISUAL Left ACT-R"])


            if buffer == "retrieval":


                # pfc_right_mean, pfc_right_error = get_raw("./rawdata/fang_pfc_R_between.csv", 2, z_transform)
                # pfc_left_mean, pfc_left_error = get_raw("./rawdata/fang_pfc_L_between.csv", 2, z_transform)
                # retrieval_right_actr_mean, retrieval_right_actr_error = get_raw("./rawdata/actr_retrieval_R_NA.csv", 2, z_transform)
                # retrieval_left_actr_mean, retrieval_left_actr_error = get_raw("./rawdata/actr_retrieval_L_NA.csv", 2, z_transform)

                pfc_right_mean = np.array([-0.14597228, -2.11154603, 0.10888775, 0.28367681, 0.1659391, -1.56455611, 0.68910539, 0.8874031, 0.84090398, -1.06401649, 0.84024013, 1.06993466])
                pfc_right_error = np.array([0.66379798, 0.64866532, 0.64767095, 0.6546017, 0.67318867, 0.6560812, 0.65306258, 0.66195463, 0.67838665, 0.65752218, 0.6552792, 0.66230866])
                pfc_left_mean = np.array([-0.91500254, -1.42168238,  0.88538117,  0.61438533, -0.74422433, -1.40215695,  1.20838027,  0.70100073, -0.56841185, -0.66338104,  1.36190375,  0.94380784])
                pfc_left_error = np.array([0.83030713, 0.84260887, 0.83833639, 0.84649913, 0.84415094, 0.83794187, 0.84074991, 0.84593709, 0.83885612, 0.84828945, 0.83947436, 0.84485611])
                retrieval_right_actr_mean = np.array([-0.31318214, -1.7251205,  -0.7654404,   0.06274823,  0.33190878, -1.38510139, -0.42887537,  1.36844243,  1.06057083, -0.28228445, 0.37081198, 1.705522])
                retrieval_right_actr_error =  np.array([0.81070688, 0.8101719, 0.79597928, 0.81413995, 0.81580376, 0.81018085, 0.80316634, 0.8203888, 0.82158118, 0.81970214, 0.80027199, 0.82447363])
                retrieval_left_actr_mean = np.array([0.09230936, -1.1936144,  -1.31399878, -0.82799938,  1.08819243, -0.95284565, -0.59756902,  0.40377756,  2.09596361,  0.55137141, -0.23456899,  0.88898185] )
                retrieval_left_actr_error =  np.array([0.98785571, 0.97603407, 0.96296534, 0.99236755, 0.99330466, 0.98308103, 0.97458447, 0.99194982, 0.99580749, 0.9900085, 0.97329274, 0.99880836])

                container = []
                container.append([pfc_right_mean, pfc_right_error, "PFC Right Fang"])
                container.append([pfc_left_mean, pfc_left_error, "PFC Left Fang"])
                container.append([retrieval_right_actr_mean, retrieval_right_actr_error, "RETRIEVAL Right ACT-R"])
                container.append([retrieval_left_actr_mean, retrieval_left_actr_error, "RETRIEVAL Left ACT-R"])




            elif buffer == "production":
                # caudate_left_mean, caudate_left_error = get_raw("./rawdata/fang_caudate_L_between.csv", 2, z_transform)
                # caudate_right_mean, caudate_right_error = get_raw("./rawdata/fang_caudate_R_between.csv", 2, z_transform)
                # atr_prod_left_mean, actr_prod_left_error = get_raw("./rawdata/actr_production_L_NA.csv", 2, z_transform)
                # atr_prod_right_mean, actr_prod_right_error = get_raw("./rawdata/actr_production_R_NA.csv", 2, z_transform)

                caudate_left_mean = np.array([-0.31922233, -1.6731295,  -0.47805088,  1.51966311,  0.07769845, -1.0907781, 0.11485281,  1.46941386,  0.33153623, -1.29092408, 0.15964573, 1.1792947])
                caudate_left_error = np.array([0.73060217, 0.72171049, 0.71666139, 0.73084428, 0.73056006, 0.72106182, 0.72051921, 0.73082795, 0.73246415, 0.72120796, 0.71953977, 0.72992674])
                caudate_right_mean = np.array([-0.12814927, -1.81083513,  0.0112812, 1.25143498,  0.18605995, -1.39317585,  0.37817312,  1.12191327,  0.13736792, -1.44785191, 0.52178921, 1.17199253])
                caudate_right_error = np.array([0.61860642, 0.60968142, 0.60488273, 0.61535777, 0.61877275, 0.60840322, 0.60607742, 0.61437298, 0.61820633, 0.60858453, 0.60596645, 0.61570081])
                atcr_prod_left_mean = np.array([-0.60759247, -1.51351558, -0.65682503,  1.49000154, -0.80018105, -0.86764865, 0.96256387, 1.25209799,  0.09945421, -1.11580561,  0.91007757,  0.8473732 ] )
                actr_prod_left_error = np.array([1.39856227, 1.39080399, 1.35320935, 1.40306926, 1.3959246,  1.37661509, 1.39997751, 1.40255479, 1.39401795, 1.37242267, 1.35009381, 1.3859465 ])
                atcr_prod_right_mean = np.array([-0.5312886,  -1.89847266, -1.12520611,  0.76495993, -0.12225018, -1.16005001 , 1.0280799,   0.77648214,  0.25357718, -0.39214406,  0.94968112 , 1.45663137] )
                actr_prod_right_error = np.array( [1.54494139, 1.53822195, 1.49970436, 1.53123446, 1.55981635, 1.54149964, 1.52084126, 1.53546117, 1.55806727, 1.54972865, 1.5043815, 1.55417838])

                container = []
                container.append([caudate_left_mean, caudate_left_error, "Caudate Left Fang"])
                container.append([caudate_right_mean, caudate_right_error, "Caudate Right Fang"])
                container.append([atcr_prod_left_mean, actr_prod_left_error, "PRODUCTION Left ACT-R"])
                container.append([atcr_prod_right_mean, actr_prod_right_error, "PRODUCTION Right ACT-R"])




            elif buffer == "goal":
                # get data
                # apfc_right_mean, apfc_right_error = get_raw("./rawdata/fang_apfc_R_between.csv", 2, z_transform)         # check goal apfc
                # acc_right_mean, acc_right_error = get_raw("./rawdata/fang_acc_R_between.csv", 2, z_transform)         # check goal acc
                # acc_actr_right_mean, acc_actr_right_error = get_raw("./rawdata/actr_goal_R_NA.csv", 2, z_transform)         # check goal acc
                # acc_actr_left_mean, acc_actr_left_error = get_raw("./rawdata/actr_goal_L_NA.csv", 2, z_transform)         # check goal acc

                # same values but saving processing time
                apfc_right_mean = np.array([0.78635276, -0.741511, -0.79064148, 0.87663237, 1.82249761, -1.00190254, -1.01664168, 0.64140844, 1.08783695, 0.01427953, -1.57263493, -0.10567602])
                apfc_right_error = np.array([2.72330271, 2.63720084, 2.73413322, 2.71152826, 2.7146019, 2.61228463, 2.67936336, 2.66873731, 2.7122128, 2.61054065, 2.70799621, 2.68022649])
                acc_right_mean = np.array([-0.32154261, -2.14949803, 0.06092221, 0.27271937, 0.05342015, -1.46179275, 0.71756787, 0.93277728, 0.86512968, -0.95489232, 0.84638294, 1.13880621])
                acc_right_error = np.array([0.60337754, 0.59071712, 0.58717578, 0.59561927, 0.61254823, 0.59790203, 0.59351176, 0.6044026, 0.61894313, 0.5994012, 0.5950822, 0.60500847])
                acc_actr_right_mean = np.array([0.79465664, 0.73420672, 0.8590016, 0.85658123, 0.8108134, 0.77153325, 0.85469579, 0.84492853, 0.80934235, 0.7551148, 0.84071622, 0.8386807, -1.16138522, -1.24074134, -1.15642268, -1.21022738, -1.0907789, -1.2521305, -1.18190348, -1.47668173])
                acc_actr_right_error = np.array([0.03285451, 0.03218221, 0.03321453, 0.03279999, 0.03311214, 0.03212455, 0.03317204, 0.03259828, 0.03307185, 0.0319738, 0.03275281, 0.03288585, 0.07807456, 0.07642832, 0.075825, 0.07661974, 0.0842545, 0.07682262, 0.08298656, 0.11178871])
                acc_actr_left_mean = np.array([-0.75474883, -1.95848588, 0.9219453, 1.02451762, -0.22842248, -0.85380988, 1.05144797, 0.82201628, -0.19204014, -1.37838689, 0.79309926, 0.75286769])
                acc_actr_left_error = np.array([0.60995768, 0.59613313, 0.61683792, 0.60779051, 0.6176279, 0.59482931, 0.6185361, 0.60464062, 0.61865695, 0.59455601, 0.60937188, 0.61158284])

                container = []
                container.append([acc_right_mean, acc_right_error, "ACC Right Fang"])
                container.append([apfc_right_mean, apfc_right_error, "APFC Right Fang"])
                container.append([acc_actr_right_mean, acc_actr_right_error, "GOAL Right ACT-R"])
                container.append([acc_actr_left_mean, acc_actr_left_error, "GOAL Left ACT-R"])


            elif buffer == "manual":
                label="Motor (ACT-R) Left"
                # manual_right_actr_mean, manual_right_actr_error = get_raw("./rawdata/actr_manual_R_NA.csv", 2, z_transform)        # check imaginal
                # manual_left_actr_mean, manual_left_actr_error = get_raw("./rawdata/actr_manual_L_NA.csv", 2, z_transform)        # check imaginal

                manual_left_actr_mean = np.array([-6.82249425e-02, -4.89164080e-01, -7.47048875e-02, 1.96078034e-01, -1.20813674e-01, -4.64428396e-01,  9.41323618e-03,  2.87364934e-01, -7.31524811e-02, -4.95463490e-01,  5.42650365e-02,  4.73761826e-01, 5.47632426e-01,  1.34309542e-01,  4.88882341e-01,  5.27497484e-01, 1.86562390e+00, -1.38278908e-03,  9.13406591e-01, -3.71090061e+00])
                manual_left_actr_error = np.array([0.31181227, 0.31160762, 0.30512535, 0.31228649, 0.31191324, 0.31186404, 0.30682331, 0.31184415, 0.31103212, 0.3104571,  0.30582556, 0.3139477, 0.31502053, 0.31954587, 0.31090549, 0.31543664, 0.34982522, 0.32253476, 0.33838319, 0.43352288])
                manual_right_actr_mean = np.array([0.21151918, -0.48414857, -0.0165865,   0.5493422,   0.32714349, -0.30979874, 0.14716814, 0.49115297, 0.22525279, -0.24387307, 0.17840435, 0.30328828, 0.41919043, -0.27671818, 0.04694543, 0.17150912,  1.5960937, -0.05527575, 0.65133378, -3.93194305])
                manual_right_actr_error = np.array([0.33085026, 0.32621069, 0.32129846, 0.32933226, 0.33400583, 0.32894318, 0.32416429, 0.33056114, 0.33347185, 0.32926098, 0.3227803, 0.32867758, 0.334194, 0.32960634, 0.32311674, 0.32929333, 0.36486102, 0.3369653, 0.35513167, 0.46987355])

                container = []
                container.append([manual_left_actr_mean, manual_left_actr_error, "MANUAL Left ACT-R"])
                container.append([manual_right_actr_mean, manual_right_actr_error, "MANUAL Right ACT-R"])

            elif buffer == "imaginal":
                # ppc_right_mean, ppc_right_error = get_raw("./rawdata/fang_ppc_R_between.csv", 2, z_transform)        # check imaginal
                # ppc_left_actr_mean, ppc_left_actr_error = get_raw("./rawdata/actr_imaginal_L_NA.csv", 2, z_transform)        # check imaginal LEFT
                # ppc_right_actr_mean, ppc_right_actr_error = get_raw("./rawdata/actr_imaginal_R_NA.csv", 2, z_transform)        # check imaginal RIGHT

                ppc_right_mean = np.array([-1.68387506, -1.29926607, -1.40143279,  0.19302189, -0.39484904, -0.28013921, -0.18250095,  1.1811858,   0.79005388,  0.88964215,  1.13042427,  1.05773513])
                ppc_right_error = np.array([0.98891095, 0.97558271, 0.99439469, 0.99126934, 0.9999333,  0.9728538, 1.00434486, 0.99548768, 0.99415743, 0.98145931, 1.00847634, 0.99636584])
                imaginal_left_actr_mean = np.array([-1.00631586, -0.47258006, -1.84703661,  0.17931731,  0.23314309,  0.25957998, -0.98380832 , 1.0855591  , 0.48844902 , 1.17084592, -0.78730949,  1.68015591])
                imaginal_left_actr_error = np.array([1.34626929 ,1.34747901 ,1.31767327, 1.35787304, 1.35748054, 1.34936158 ,1.32778481, 1.36122404, 1.37364238 ,1.35688778, 1.34231768, 1.36384092])
                imaginal_right_actr__mean = np.array([-0.17941497, -0.78205773 ,-1.19986804 , 0.83588133 , 0.66601563 ,-1.01773519 ,-1.14510649 , 1.11775991 , 1.06688226  ,0.05417135, -1.09866059 , 1.68213253] )
                imaginal_right_actr_error = np.array([1.3688221,  1.3724674,  1.35320887, 1.38446107, 1.38997279, 1.37596576, 1.36271593, 1.38627171, 1.39325925, 1.38072511, 1.37277067, 1.38766851])

                container = []
                container.append([ppc_right_mean, ppc_right_error, "PPC Right Fang"])

                container.append([imaginal_left_actr_mean, imaginal_left_actr_error, "IMAGINAL Left ACT-R"])
                container.append([imaginal_right_actr__mean, imaginal_right_actr_error, "IMAGINAL Right ACT-R"])



            for c in container:

                if plot:

                    axes = plt.gca()

                    axes.errorbar(np.arange(len(c[0])) *1000, c[0], c[1], color="black", linestyle='None', marker='None', capsize=2,  alpha=0.8)
                    axes.plot(np.arange(len(c[0])) *1000, c[0], label=c[2], color="black",  marker='o', linewidth=2.0, mew=3.0)
                    # plt.fill_between(np.arange(len(c[0])) * 1000 , c[0]- c[1], c[0]+ c[1], color="black", alpha=0.2)



                # cut arrays to make them equal size
                c[0] = c[0][0:min(len(c[0]), len(pred_means_z))]
                short_pred_means_z = pred_means_z[0:min(len(c[0]), len(pred_means_z))]


                ken_tau, ken_p_value = stats.kendalltau(c[0], short_pred_means_z)

                pear_tau, pear_p_value = stats.pearsonr(c[0], short_pred_means_z)

                spear_rho, spear_p_value = stats.spearmanr(c[0], short_pred_means_z)

                correlation_counter += 1
                correlation_sum += spear_rho

                if plot:
                    spearstring  = 'Spearman:  r=' + shorten(str(spear_rho)) + "    p=" + shorten(str(spear_p_value))  + "\n"
                    kenstring    = 'Kendall:   t=' + shorten(str(ken_tau)) + "    p=" + shorten(str(ken_p_value)) + "\n"
                    pearsonsring = 'Pearson:   r=' + shorten(str(pear_tau)) + "    p=" + shorten(str(pear_p_value))




                    plt.annotate(spearstring + kenstring + pearsonsring, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top')






                    args = sys.argv


                    argsstring = "runs:    " + str(runs) + "\n" + "model:  " + model + "\n" + "ans:     " + str(ans) + "\n" + "RT:      " + str(rt) + "\n" + "LF:      " + str(lf) + "\n"
                    plt.annotate(argsstring, (0,0), (200, -30), xycoords='axes fraction', textcoords='offset points', va='top')
                    argsstring = "scale :    " + str(bold_scale) + "\n" + "neg scale:   " + str(neg_bold_scale) + "\n" + "exp:           " + str(bold_exp) + "\n" + "neg exp:      " + str(neg_bold_exp) + "\n" + "positive:     " + str(bold_positive) + "\n"  + "negative:     " + str(bold_negative)
                    plt.annotate(argsstring, (0,0), (300, -30), xycoords='axes fraction', textcoords='offset points', va='top')



                    axes.plot(np.arange(len(short_pred_means_z)) *1000, short_pred_means_z, label=buffer[0].upper()+ buffer[1:] + " ACT-R 7 prediction", marker="o", color="red", linewidth=2.0, mew=3.0)



                    # plot
                    plt.xlabel('Time (ms)')
                    plt.ylabel('BOLD response (z-transformed)')


                    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

                    if z_transform:
                        axes.set_ylim([min(0, -4),  max(0.6, 4)])
                    else:
                        axes.set_ylim([0,0.6])

                    axes.set_xlim([0,(len(short_pred_means_z)-1)*1000])


                    axes.tick_params(bottom=True, top=True, left=True, right=True)


                    plt.subplots_adjust(bottom=0.2)
                    plt.tight_layout()
                    plt.savefig(output_path + "/bold_z_trans_" + c[2].replace(" ", "_") + ".png", dpi=250)

                    plt.clf()
                    plt.close('all')









    print(correlation_sum, correlation_counter, correlation_sum/correlation_counter)


    return correlation_sum/correlation_counter
    return spear_rho






def tolerant_mean(arrs, z_transform):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True

    for idx, l in enumerate(arrs):
        if z_transform:
            arr[:len(l),idx] = z_transformation(l)
        else:
            arr[:len(l),idx] = l

    return arr.mean(axis = -1), arr.std(axis=-1), lens



def get_bold(boldfile):
    bold = []

    with open(boldfile) as f:
        line = f.readline()

        if "#|Warning" in line:
            line = f.readline()
            warnings += 1

        line = line.strip().split(" ")
        while "" in line:
            line.remove("")


        bufferstring = line

        line = f.readline()


        while len(line) > 1:
            if "#|Warning" in line:
                warnings += 1

                line = f.readline()

                continue
            line = line.strip().split(" ")
            while "" in line:
                line.remove("")


            line = [float(i) for i in line]
            bold.append(line)
            line = f.readline()
    return np.array(bold), np.array(bufferstring)


def shorten(val):
    if "e-" in val:
        return  shorten(str('%f'%float(val)))
    if "." in val:
        temp = val.split(".")
        if len(temp[1])>5:
            return temp[0] + "." + temp[1][0:4]
    return val




