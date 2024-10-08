from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_recommendations(data, scores, k=5):
    scores = scores.sort_values("score", ascending=False)

    fig, axs = plt.subplots(k, 1, figsize=(10, 20), tight_layout=True)
    fig.suptitle("Estimations from " + scores.iloc[0]["algorithm"])
    for i in range(k):
        topK = scores.iloc[i]
        dt = data.execute(topK["condition"])[["feature_0", "feature_1"]]
        topK["condition"] = topK["condition"].replace("AND", "\n AND")
        dt = pd.DataFrame(dt, columns=["feature_0", "feature_1"])

        sns.barplot(x=["T0", "T1"], y=[topK["T0"], topK["T1"]], ax=axs[i])
        axs[i].set_title("K=" + str(i + 1))
        axs[i].set_yticks(np.arange(0, 8, 1))
        axs[i].set_xlabel("treatment")
        axs[i].set_ylabel("outcome")

        outer_text = f"{topK['condition']} \n\n Size: {dt.shape[0]} \n Est CATE: {str(topK['t_est'])} \n overlap: {str(topK['overlap'])}\n Score: 0.6*{str(topK['norm_cate'])} + (1-0.6)*{str(topK['overlap'])}  =  {str(round(topK['score'],2))}"
        axs[i].annotate(
            outer_text,
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(1.02, 1),
            textcoords="axes fraction",
            ha="left",
            va="top",
        )


def visualize_recommendations2(data, scores, k=5):
    scores = scores.sort_values("score", ascending=False)

    fig, axs = plt.subplots(k, 1, figsize=(10, 20), tight_layout=True)
    fig.suptitle("Recommendations from " + scores.iloc[0]["algorithm"])
    for i in range(k):
        topK = scores.iloc[i]
        dt = data.execute(topK["condition"])[["feature_0", "feature_1"]]
        topK["condition"] = topK["condition"].replace("AND", "\n AND")

        dt = pd.DataFrame(dt, columns=["feature_0", "feature_1"])

        sns.scatterplot(x="feature_0", y="feature_1", data=dt, ax=axs[i])

        axs[i].set_title("K=" + str(i + 1))
        axs[i].set_xticks(np.arange(-4, 4, 1))
        # axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_yticks(np.arange(-4, 4, 1))
        # axs[i].set_xlabel("treatment")
        # axs[i].set_ylabel("outcome")

        outer_text = f"{topK['condition']} \n\n Size: {dt.shape[0]} \n Est CATE: {str(topK['t_est'])} \n overlap: {str(topK['overlap'])}\n Score: 0.6*{str(topK['norm_cate'])} + (1-0.6)*{str(topK['overlap'])}  =  {str(round(topK['score'],2))}"
        axs[i].annotate(
            outer_text,
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(1.02, 1),
            textcoords="axes fraction",
            ha="left",
            va="top",
        )
