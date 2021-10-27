import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib


def evaluate(        output_path="./plots", filename="actr7"):
    bold_table = np.zeros([13, 60])
    bold_table_divider = np.zeros([13, 60])
    boldfiles = [x[0] for x in os.walk("./log/bolds")]

    inslen = 0
    arrays = []
    longcount = 0

    for boldfile in sorted(boldfiles):
        if boldfile.count("/") < 5:
            continue

        if "training" in boldfile.lower():
            continue



        bold_result, buffer_list = get_bold(os.path.abspath(boldfile + "/bold-response.dat"))
        inslen = inslen + len(bold_result)

        if len(bold_result) > 32:
            longcount = longcount + 1


        permutation = np.argsort(buffer_list)


        for bs_ix in range(len(bold_result)):
            bold_result[bs_ix] = bold_result[bs_ix, permutation]
        buffer_list = buffer_list[permutation]




        arrays.append(bold_result)
    print("insgesamt länge ", inslen)
    print(longcount)

    matplotlib.use("agg", force=True)

    plt.clf()


    plt.tight_layout()

    for b in range(len(buffer_list)):
        arrr = []

        buffer = buffer_list[b]



        for a in arrays:
            nparray = np.array(a)

            arrr.append(nparray[:, b])

        axes = plt.gca()



        buffer = buffer.lower()

        if buffer in ["time", "temporal", "aural-location", "vocal", "visual-location", "production", "aural", "visual", "retrieval"]:
            continue

        y, error = tolerant_mean(arrr)


        buffer = buffer.lower()
        if buffer == "retrieval":
            marker = "+"
            color = "black"
        elif buffer == "goal":
            marker = "4"
            color = "pink"
        elif buffer == "manual":
            marker = "v"
            color = "y"
        elif buffer == "visual":
            marker = "s"
            color = "b"
        elif buffer == "aural":
            marker = "s"
            color = "purple"
        elif buffer == "imaginal":
            marker = "o"
            color = "g"
        else:
            print(buffer)
            1/0

        axes.plot(np.arange(len(y))/2, y, label=buffer.lower(), marker=marker, color=color)
        plt.fill_between(np.arange(len(y))/2, y - error, y + error, color=color, alpha=0.2)


    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD response')
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

    axes.set_ylim([0, 1])
    axes.set_xlim([0, 25])

    plt.savefig(output_path + "/" + filename + "graph.png")

    plt.clf()



def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


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







evaluate()
