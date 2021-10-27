"""
Reads the information from the file modelresults.dat and converts it into numpy arrays. At the same time, it calculates the average reasoning time for the different trail types - i.e. the time the model needs to respond to the stimuli of a conclusion. It is then differentiated into trails:
The trail where the model accepts the conclusion: Regardless of whether the conclusion was correct.
The trail where model rejects conclusion: Regardless of whether the conclusion was wrong
The trail where the conclusion is correct: A<B, B<C, C>A
The trail where the conclusion is wrong: A<B, B<C, C<A
The trail where conclusion cannot be solved: A<B, A<C, C<B
Trails where the model has correctly accepted
Trails where model correctly rejected because conclusions were wrong
Trails where the model has correctly rejected because it was not solvable
Trails where the conclusion was correct but was rejected (Error Type I)
Trails where the conclusion was wrong but was accepted (Error Type II)
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib

def get_answers(run_amount=None, model=None):
    all_trails = []

    trails_model_accepts_conclusion = []
    trails_model_declines_conclusion = []

    trails_with_correct_conclusion = []
    trails_with_non_correct_conclusion = []
    trails_with_indeterminate_conclusions = []

    model_predicted_correct_accept = []
    model_predicted_correct_decline = []
    model_predicted_correct_indeterminate = []
    model_error_type_I = []
    model_error_type_II = []

    accept = "r"
    decline = "f"

    test = 0
    run = -1

    meta = []


    with open("log/model/" + model + ".dat") as f:
        line = f.readline()
        line = f.readline()

        bad = False

        data = []
        counter = 0
        while len(line) > 1:
            counter = counter +1
            line = line.split(" ")
            while "" in line:
                line.remove("")
            line[-1] = line[-1].replace("\n", "")

            if "training" in line[2].lower():
                line = f.readline()
                continue



            line[0] = int(line[0])
            line[1] = int(line[1])
            line[2] = int(line[2])
            line[3] = int(line[3])
            line[4] = int(line[4])

            # get meta data of last row, this has to be changed if values change within the experiment
            meta = line[7:]


            data.append(line)
            line = f.readline()

            continue

        data = np.array(data, dtype = 'object')


        ms_index = 3
        response_index = 5
        expected_index = 6

        runs = max(data[:,0])


        broken_runs = np.unique(data[np.where(data[:,5] == "NA"),0])

        # remove a complete run if there is at leasr one single task that could not return something
        bad_indices = np.flatnonzero(np.isin( data[:,0], broken_runs))
        data = np.delete(data, bad_indices, axis=0)


        all_trails = data[:, ms_index]


        # genereal response was acceptance or decline
        trails_model_accepts_conclusion = data[np.where(data[:, response_index] == accept), ms_index][0]
        trails_model_declines_conclusion = data[np.where(data[:, response_index] == decline), ms_index][0]




        # generel the expected respone was accept or decline
        trails_with_correct_conclusion = data[np.where(data[:, expected_index] == accept), ms_index][0]
        trails_with_non_correct_conclusion = data[np.where(data[:, expected_index] == decline), ms_index][0]







        # the model CORRECTLY accepted a CORRECT conclusion
        model_predicted_correct_accept = data[np.where(np.logical_and(data[:, expected_index] == accept, data[:, response_index] == accept)), ms_index][0]
        # the model CORRECTLY declines a INCORRECT conclusion
        model_predicted_correct_decline = data[np.where(np.logical_and(data[:, expected_index] == decline, data[:, response_index] == decline)), ms_index][0]
        # error type I
        model_error_type_I = data[np.where(np.logical_and(data[:, expected_index] == accept, data[:, response_index] == decline)), ms_index][0]
        # error type II
        model_error_type_II = data[np.where(np.logical_and(data[:, expected_index] == decline, data[:, response_index] == accept)), ms_index][0]








    trails_lengths = [len(all_trails),
              len(trails_model_accepts_conclusion),
              len(trails_model_declines_conclusion),

              len(trails_with_correct_conclusion),
              len(trails_with_non_correct_conclusion),

              len(model_predicted_correct_accept),
              len(model_predicted_correct_decline),
              len(model_error_type_I),
              len(model_error_type_II)]
    trails_avg_times = [get_avt_time(all_trails),
                    get_avt_time(trails_model_accepts_conclusion),
                    get_avt_time(trails_model_declines_conclusion),

                    get_avt_time(trails_with_correct_conclusion),
                    get_avt_time(trails_with_non_correct_conclusion),

                    get_avt_time(model_predicted_correct_accept),
                    get_avt_time(model_predicted_correct_decline),
                    get_avt_time(model_error_type_I),
                    get_avt_time(model_error_type_II)]

    trails_header = ["All Trails",
              "Accepted",
              "Declined",
              "Correct Conclusion",
              "Not Correct Conclusion",
              "prediction correct accept",
              "prediction correct decline",
              "error type I",
              "error type II"]








    # print(all_trails)

    # print(len(all_trails), len(model_predicted_correct_accept) + len(model_predicted_correct_decline) + len(model_error_type_I) + len(model_error_type_II))

    if trails_lengths[0] == 0:
        # all crashed
        return trails_lengths, trails_avg_times, trails_header, broken_runs, runs, str(0), meta

    else:
        return trails_lengths, trails_avg_times, trails_header, broken_runs, runs, shorten(str((trails_lengths[5] + trails_lengths[6]) / (trails_lengths[0]) * 100)), meta

def get_avt_time(trail_list):
    if len(trail_list) == 0:
        return  [0, 0]

    return [np.mean(trail_list), np.std(trail_list)]



def shorten(val):
    if "." in val:

        temp = val.split(".")
        if len(temp[1])>2:
            return temp[0] + "." + temp[1][0:2]
        else:
            return val

def plot_answers(
        trails_length=None,
        trails_avg_times=None,
        trails_header=None,
        output_path=None,
        add_human_data=True):

    matplotlib.use("agg", force=True)

    x = np.arange(1)  # the label locations

    fig, ax = plt.subplots()



    width = 0.25  # the width of the bars
    rects0 = ax.bar(x - 2 * width, trails_avg_times[4][0], width, label=trails_header[4] + " std: " + str(trails_avg_times[4][1]+0.0000001)[0:4])
    rects1 = ax.bar(x - 1 * width, trails_avg_times[5][0], width, label=trails_header[5] +  " std: " + str(trails_avg_times[5][1]+0.0000001)[0:4])
    rects2 = ax.bar(x, trails_avg_times[6][0], width, label=trails_header[6] + " std:" + str(trails_avg_times[6][1]+0.0000001)[0:4])
    rects3 = ax.bar(x + width,trails_avg_times[7][0], width, label=trails_header[7] + " std: " + str(trails_avg_times[7][1]+0.0000001)[0:4])
    rects4 = ax.bar(x + 2 * width, trails_avg_times[8][0], width, label=trails_header[8] + " std: " + str(trails_avg_times[8][1]+0.0000001)[0:4])

    ax.hlines(trails_avg_times[0][0], -0.6, 0.6,
              colors='k', linestyles='solid', label='avergage std:' + str(trails_avg_times[0][1]+0.0000001)[0:4])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('milliseconds')
    ax.set_xticks(x)
    ax.set_xticklabels(["Timing"])
    ax.legend()


    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3, fancybox=True, shadow=True)
    # plt.savefig("plots/times.png", bbox_inches='tight')

    plt.clf()

    # Make data: I have 3 groups and 7 subgroups
    assert trails_length[0] == trails_length[5] + trails_length[6] + trails_length[7] + trails_length[8]

    if(trails_length[0] == 0):
        print("Not a single run was good")

    std = np.std(np.concatenate([np.ones(trails_length[0]- (trails_length[5] + trails_length[6])), np.zeros(  (trails_length[5] + trails_length[6]))]) )

    group_names = ['Correct ' +  str(((trails_length[5] + trails_length[6]) / (trails_length[0]) * 100) + 0.000000000001)[0:4] + " %" + "    std: " + str(std+0.0000000)[0:4], 'Incorrect']
    group_size = [
        trails_length[5] +
        trails_length[6],
        trails_length[7] +
        trails_length[8]]

    if trails_length[5] > 0:
        one = ["correct"]
    else:
        one = []

    if trails_length[6] > 0:
        two = ["declined"]
    else:
        two = []

    if trails_length[7] > 0:
        three = ["Error I " +  str(trails_length[7] / trails_length[0] *100  +0.00000000000000000000001)[0:4] + " %"]
    else:
        three = []

    if trails_length[8] > 0:
        four = ["Error II " + str(trails_length[8] / trails_length[0] * 100 +0.00000000000000001)[0:4] + " %"]
    else:
        four = []

    subgroup_names = one + two + three + four

    if trails_length[5] > 0:
        one = [trails_length[5]]
    else:
        one = []

    if trails_length[6] > 0:
        two = [trails_length[6]]
    else:
        two = []

    if trails_length[7] > 0:
        three = [trails_length[7]]
    else:
        three = []

    if trails_length[8] > 0:
        four = [trails_length[8]]
    else:
        four = []

    subgroup_size = one + two + three + four

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Reds]

    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(
        group_size, radius=1.3, labels=group_names, colors=[
            a(0.6), b(0.6)])
    plt.setp(mypie, width=0.3, edgecolor='white')

    if trails_length[5] > 0:
        one = [a(0.5)]
    else:
        one = []

    if trails_length[6] > 0:
        two = [a(0.4)]
    else:
        two = []

    if trails_length[7] > 0:
        three = [b(0.5)]
    else:
        three = []

    if trails_length[8] > 0:
        four = [b(0.4)]
    else:
        four = []

    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3 -
                       0.3, labels=subgroup_names, labeldistance=0.7, colors=one +
                       two +
                       three +
                       four)
    plt.setp(mypie2, width=0.4, edgecolor='white')
    plt.margins(0, 0)

    if trails_length[5] > 0:
        one = ["Accept correct Conclusion"]
    else:
        one = []

    if trails_length[6] > 0:
        two = ["Denied incorrect Conclusion"]
    else:
        two = []

    if trails_length[7] > 0:
        three = ["Denied correct Conclusion"]
    else:
        three = []

    if trails_length[8] > 0:
        four = ["Accepted incorrect Conclusion"]
    else:
        four = []

    subgroup_names_legs = one + two + three + four

    # plt.legend(loc=(0.9, 0.1))
    handles, labels = ax.get_legend_handles_labels()

    # ax.legend(handles[3:], subgroup_names_legs, loc=(0.9, 0.1))

    # show it
    # plt.savefig("plots/answers.png", bbox_inches='tight')







