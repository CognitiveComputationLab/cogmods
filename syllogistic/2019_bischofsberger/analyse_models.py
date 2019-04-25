""" A script to verify the responses of recreated models """

import ccobra
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

from modular_models.models.ccobra_models import CCobraAtmosphere, CCobraAbstractAtmosphere, CCobraMatching, \
    CCobraGeneralizedMatching, CCobraIllicitConversion, CCobraPHM, CCobraGeneralizedPHM, CCobraPSYCOP,\
    CCobraMentalModels, CCobraVerbalModels, CCobraLogicallyValidLookup
from modular_models.util import sylutil

PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/"))
DATA_PLOT_DIR = os.path.join(PLOT_DIR, "data_plots/")
MODEL_STATS_PLOT_DIR = PLOT_DIR
PARAM_PLOT_DIR = os.path.join(PLOT_DIR, "param_plots/")

TRAINING_DATA_CSV = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/Veser2018.csv"))
TEST_DATA_CSV = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/Ragni2016.csv"))


def score(df):
    return df["correct"].sum() / len(df.index)


def plot_model_performance_stats(df_model):
    model = df_model["model"].iloc[0]
    correct = df_model["correct"].sum() / len(df_model.index)

    correct_by_fig = []
    for figure in ["1", "2", "3", "4"]:
        df_fig = df_model[df_model["syl_figure"] == figure]
        correct_by_fig.append(df_fig["correct"].sum() / len(df_fig.index))

    correct_by_mood = []
    for mood in ["A", "I", "E", "O"]:
        df_mood = df_model[df_model["syl_mood_" + mood]]
        correct_by_mood.append(df_mood["correct"].sum() / len(df_mood.index))

    correct_by_validity = []
    for tf in [True, False]:
        df_validity = df_model[df_model["syl_valid"] == tf]
        correct_by_validity.append(df_validity["correct"].sum() / len(df_validity.index))

    correct_by_mood_eq = []
    for tf in [True, False]:
        df_meq = df_model[df_model["syl_mood_equal"] == tf]
        correct_by_mood_eq.append(df_meq["correct"].sum() / len(df_meq.index))

    fig, axes = plt.subplots(3, 2)

    n_nvc_resp = len(df_model[df_model["response_enc"] == "NVC"].index)
    n_non_nvc_resp = len(df_model[df_model["response_enc"] != "NVC"].index)

    bars1 = [
        float(df_model[(df_model["response_enc"] != "NVC") & (df_model["prediction_enc"] == "NVC")][
                  "1"].sum()) / n_non_nvc_resp,
        float(df_model[(df_model["mood_correct"]) & (df_model["response_enc"] != "NVC")][
                  "mood_correct"].sum()) / n_non_nvc_resp,
        float(df_model[(df_model["order_correct"]) & (df_model["response_enc"] != "NVC")][
                  "order_correct"].sum()) / n_non_nvc_resp,
        float(df_model[(df_model["mood_and_order_correct"]) & (df_model["response_enc"] != "NVC")][
                  "mood_and_order_correct"].sum()) / n_non_nvc_resp,
        float(df_model[(df_model["mood_and_order_incorrect"]) & (df_model["response_enc"] != "NVC")][
                  "mood_and_order_incorrect"].sum()) / n_non_nvc_resp,
    ]

    bars2 = [
        df_model[(df_model["response_enc"] != "NVC")]["correct"].sum() / n_non_nvc_resp,
        df_model[(df_model["response_enc"] == "NVC")]["correct"].sum() / n_nvc_resp,
    ]

    axes[2, 0].bar(range(5), bars1, tick_label=["nvc predicted\n(non-nvc resp.)", "correct mood\n(non-nvc resp.)",
                                                "correct ac-order\n(non-nvc resp.)",
                                                "both correct\n(non-nvc resp.)", "both incorrect\n(non-nvc resp.)"])
    axes[2, 1].bar(range(2), bars2, tick_label=["correct (non-nvc resp.)", "correct (nvc resp.)"])
    axes[2, 1].plot(range(2), [correct] * 2, color="r")
    [ax.yaxis.grid(True) for i, _ in enumerate(axes) for ax in axes[i]]
    [plt.sca(ax) and plt.xticks(fontsize=13) for i, _ in enumerate(axes) for ax in axes[i]]
    fig.set_size_inches(8, 6)

    for i, _ in enumerate(axes):
        for ax in axes[i]:
            plt.sca(ax)
            plt.xticks(fontsize=5)

    fig.suptitle(model + ": " + str(correct))

    axes[0, 0].bar(x=range(4), height=correct_by_fig, tick_label=["1", "2", "3", "4"])
    axes[1, 0].bar(x=range(4), height=correct_by_mood, tick_label=["A", "I", "E", "O"])
    axes[0, 1].bar(x=range(2), height=correct_by_validity, tick_label=["valid", "invalid"])
    axes[1, 1].bar(x=range(2), height=correct_by_mood_eq, tick_label=["equal", "different"])
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        n = 4 if j == 0 else 2
        axes[i, j].plot(range(n), [correct] * n, color="r")
    plt.savefig(os.path.join(MODEL_STATS_PLOT_DIR, model + ".png"), dpi=500)
    print("Saved", os.path.join(MODEL_STATS_PLOT_DIR, model + ".png"))


def plot_syllogistic_statistics(dataset_test, m_logic):
    agg_data = sylutil.aggregate_data(dataset_test)
    for syllogism in agg_data:
        y_logic = m_logic.cached_prediction(syllogism)
        D = agg_data[syllogism]
        plt.figure()
        plt.title(syllogism)
        plt.bar(range(len(D)), list(D.values()), align="center", color="red", label="data")
        plt.bar(range(len(y_logic)), y_logic, align="edge", color="blue", label="logic")
        plt.xticks(range(len(D)), list(D.keys()))
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(DATA_PLOT_DIR, syllogism + ".png"), dpi=500)
        print("Saved", os.path.join(DATA_PLOT_DIR, syllogism + ".png"))
        plt.close()


def main():
    random.seed(12345)

    data_frame_training = pd.read_csv(TRAINING_DATA_CSV, header=0)
    dataset_training = sylutil.persubjectify(data_frame_training)

    data_frame_test = pd.read_csv(TEST_DATA_CSV, header=0)
    dataset_test = sylutil.persubjectify(data_frame_test)

    m_logic = CCobraLogicallyValidLookup()

    print("Making data plots...")
    plot_syllogistic_statistics(dataset_test, m_logic)

    # models to be tested
    print("Initializing models...")
    models = [
              CCobraAtmosphere(),
              CCobraAbstractAtmosphere(),
              CCobraMatching(),
              CCobraGeneralizedMatching(),
              CCobraIllicitConversion(),
              CCobraPHM(),
              CCobraGeneralizedPHM(),
              CCobraPSYCOP(),
              CCobraMentalModels(),
              CCobraVerbalModels(),
              CCobraLogicallyValidLookup(),
             ]

    df_prediction_data = pd.DataFrame(columns=["model", "pre_trained", "adapted", "subject_id", "syllogism_enc",
                                               "prediction_enc", "response_enc"])

    data_features = []
    for i_subj, subject_data in enumerate(dataset_test):
        data_features.append([])
        for i_item, item_data in enumerate(subject_data):
            data_features[-1].append({"item": item_data["item"], "response": item_data["response"], "subj_id": i_subj,
                    "syl_enc": ccobra.syllogistic.encode_task(item_data["item"].task),
                    "resp_enc": ccobra.syllogistic.encode_response(item_data["response"], item_data["item"].task)})

    for model in models:
        print("Collecting data for model", model.name + "...")
        for pre_training, adaption in [(True, True)]:  # (False, False), (True, False), (True, True)]:
            if pre_training:
                model.pre_train(dataset_training)
            for subject_data in data_features:
                model_copy = copy.deepcopy(model)
                model_copy.start_participant()
                for item_data in subject_data:
                    prediction_enc = ccobra.syllogistic.encode_response(model_copy.predict(item_data["item"]), item_data["item"].task)

                    df_prediction_data = df_prediction_data.append({"model": model_copy.name,
                                                                    "pre_trained": pre_training,
                                                                    "adapted": adaption,
                                                                    "subject_id": item_data["subj_id"],
                                                                    "syllogism_enc": item_data["syl_enc"],
                                                                    "prediction_enc": prediction_enc,
                                                                    "response_enc": item_data["resp_enc"]},
                                                                    ignore_index=True)

                    if adaption:
                        model_copy.adapt(item_data["item"], item_data["response"])

    # counter
    df_prediction_data["1"] = df_prediction_data.apply(lambda row: 1, axis=1)

    # add item features
    df_prediction_data["syl_valid"] = df_prediction_data.apply(lambda row: m_logic.model.syllogism_is_valid(row["syllogism_enc"]), axis=1)
    df_prediction_data["syl_mood_A"] = df_prediction_data.apply(lambda row: "A" in row["syllogism_enc"], axis=1)
    df_prediction_data["syl_mood_I"] = df_prediction_data.apply(lambda row: "I" in row["syllogism_enc"], axis=1)
    df_prediction_data["syl_mood_E"] = df_prediction_data.apply(lambda row: "E" in row["syllogism_enc"], axis=1)
    df_prediction_data["syl_mood_O"] = df_prediction_data.apply(lambda row: "O" in row["syllogism_enc"], axis=1)
    df_prediction_data["syl_mood_equal"] = df_prediction_data.apply(lambda row: row["syllogism_enc"][0] == row["syllogism_enc"][1], axis=1)
    df_prediction_data["syl_figure"] = df_prediction_data.apply(lambda row: row["syllogism_enc"][2], axis=1)

    # add response features
    df_prediction_data["resp_nvc"] = df_prediction_data.apply(lambda row: "E" in row["response_enc"] == "NVC", axis=1)

    # add prediction features
    df_prediction_data["pred_nvc"] = df_prediction_data.apply(lambda row: "E" in row["prediction_enc"] == "NVC", axis=1)

    # add correctness flags
    df_prediction_data["correct"] = df_prediction_data.apply(lambda row: row["response_enc"] == row["prediction_enc"], axis=1)
    df_prediction_data["mood_correct"] = df_prediction_data.apply(lambda row: row["response_enc"][0] == row["prediction_enc"][0], axis=1)
    df_prediction_data["order_correct"] = df_prediction_data.apply(lambda row: row["response_enc"][1:] == row["prediction_enc"][1:], axis=1)
    df_prediction_data["mood_and_order_correct"] = df_prediction_data.apply(lambda row: row["mood_correct"] and row["order_correct"], axis=1)
    df_prediction_data["mood_and_order_incorrect"] = df_prediction_data.apply(lambda row: not row["mood_correct"] and not row["order_correct"], axis=1)

    for model in df_prediction_data["model"].unique():
        df_model = df_prediction_data[df_prediction_data["model"] == model]
        print("Plotting stats for model", model + "...")
        plot_model_performance_stats(df_model)

    for model in models:
        print("Plotting parameter stats for model", model.name + "...")
        try:
            if len(model.model.params) == 0:
                continue
        except:
            continue
        score_by_param = []
        for params in model.configurations:
            correct = 0
            n_items = 0
            score_by_param.append([params, 0])
            model.model.set_params(params)
            for subject_data in dataset_training:
                for item_data in subject_data:
                    n_items += 1
                    item = item_data["item"]
                    response = item_data["response"]
                    response_enc = ccobra.syllogistic.encode_response(response, item.task)
                    prediction_enc = ccobra.syllogistic.encode_response(model.predict(item), item.task)
                    if response_enc == prediction_enc:
                        correct += 1

            score_by_param[-1][1] = correct / n_items

#        sorted_scores = sorted([score_by_param], key=lambda el: el[1])[0]

        scoren = {parameter: {} for parameter in model.model.generate_param_configurations()[0]}
        for parameter in model.model.generate_param_configurations()[0]:
            for (config, scr) in score_by_param:
                value = config[parameter]
                if value not in scoren[parameter]:
                    scoren[parameter][config[parameter]] = [0, 0]
                scoren[parameter][value] = [sum(x) for x in zip(scoren[parameter][value], [scr, 1])]
            for value in scoren[parameter]:
                scoren[parameter][value] = scoren[parameter][value][0] / scoren[parameter][value][1]

        plt.figure()
        x = []
        y = []
        w = []
        labels = []
        labels_inner = []
        for i, p in enumerate(scoren):
            labels.append(p)
            for j, pval in enumerate(scoren[p]):
                x.append(i - 0.45 + 0.9*j/len(scoren[p]) + 0.01)
                y.append(scoren[p][pval])
                w.append(0.9/len(scoren[p]) - 0.02)
                labels_inner.append(pval)
        plt.bar(x=x, height=y, width=w, align="edge", color="blue", label="logic")
        plt.xticks(range(len(labels)), list(labels))

        rects = plt.gca().patches
        for rect, ll in zip(rects, labels_inner):
            height = rect.get_height()
            plt.text(x=rect.get_x() + rect.get_width()/2, y=height+0.005, s=ll,
                     ha='center', va='bottom', fontsize=6)
        plt.title(model.name)
        plt.savefig(os.path.join(PARAM_PLOT_DIR, model.name + ".png"), dpi=500)
        print("Saved", os.path.join(PARAM_PLOT_DIR, model.name + ".png"))
        plt.close()


if __name__ == "__main__":
    main()
