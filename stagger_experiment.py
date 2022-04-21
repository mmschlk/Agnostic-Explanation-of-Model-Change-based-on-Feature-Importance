# -*- coding: utf-8 -*-
from changeExplainer import changeExplainer
from river import metrics, synth, ensemble, linear_model, neighbors, preprocessing, stream
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tikzplotlib

def run_experiment(run_id, n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau):
    oh = preprocessing.OneHotEncoder()
    seed_value = np.random.randint(1000)
    stream_1 = synth.STAGGER(0, balance_classes=False, seed=seed_value)
    stream_2 = synth.STAGGER(2, balance_classes=False, seed=seed_value)
    stream = synth.ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        position=drift_position,
        width=drift_width
    )

    feature_list = ['size','color','shape']

    arf_accuracy = metrics.Accuracy()
    lm_accuracy = metrics.Accuracy()
    ibl_accuracy = metrics.Accuracy()

    arf_model = ensemble.AdaptiveRandomForestClassifier(seed=seed_value)
    arf_explainer = changeExplainer(arf_model, "Adaptive Random Forest", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)
    lm_model = linear_model.Perceptron()
    lm_explainer = changeExplainer(lm_model, "Perceptron Classifier", len_explanation_interval, adwin_delta, tau,
                                   seed_value,feature_list)
    ibl_model = neighbors.SAMKNNClassifier()
    ibl_explainer = changeExplainer(ibl_model, "SAMKNN Classifier", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)


    for (n, (x,y)) in enumerate(stream):
        oh.learn_one(x)
        if n>200:
            break

    for (n, (x, y)) in enumerate(stream):
        x=oh.transform_one(x)
        # Prediction
        y_pred_arf = arf_model.predict_one(x)
        y_pred_lm = lm_model.predict_one(x)
        y_pred_ibl = ibl_model.predict_one(x)
        # Accuracy Update
        arf_accuracy.update(y, y_pred_arf)
        lm_accuracy.update(y, y_pred_lm)
        ibl_accuracy.update(y, y_pred_ibl)
        # Learning
        arf_model.learn_one(x, y)
        lm_model.learn_one(x, y)
        ibl_model.learn_one(x, y)
        # Explaining
        arf_explainer.explain_sample(x, y, y_pred_arf)
        lm_explainer.explain_sample(x, y, y_pred_lm)
        ibl_explainer.explain_sample(x, y, y_pred_ibl)
        if n > n_stream_length:
            break
        arf_explainer.explanations["Total Accuracy"] = arf_accuracy.get()
        lm_explainer.explanations["Total Accuracy"] = lm_accuracy.get()
        ibl_explainer.explanations["Total Accuracy"] = ibl_accuracy.get()
    results = pd.concat([arf_explainer.explanations, lm_explainer.explanations, ibl_explainer.explanations])
    results["run_id"] = run_id
    return results


def plot_accuracy(n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau,
                  seed_value=43, window_size=50):
    results = []
    oh = preprocessing.OneHotEncoder()
    stream_1 = synth.STAGGER(0, balance_classes=False, seed=seed_value)
    stream_2 = synth.STAGGER(2, balance_classes=False, seed=seed_value)
    stream = synth.ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        position=drift_position,
        width=drift_width,
        seed=seed_value
    )
    feature_list = ['size','color','shape']
    arf_rolling_accuracy = metrics.Rolling(metrics.Accuracy(), window_size=window_size)
    arf_model = ensemble.AdaptiveRandomForestClassifier(seed=seed_value)
    arf_explainer = changeExplainer(arf_model, "Adaptive Random Forest", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)

    for (n, (x,y)) in enumerate(stream):
        oh.learn_one(x)
        if n>200:
            break

    for (n, (x, y)) in enumerate(stream):
        x=oh.transform_one(x)
        # Prediction
        y_pred_arf = arf_model.predict_one(x)
        # Accuracy Update
        arf_rolling_accuracy.update(y, y_pred_arf)
        results.append(arf_rolling_accuracy.get())
        # Learning
        arf_model.learn_one(x, y)
        # Explaining
        arf_explainer.explain_sample(x, y, y_pred_arf)
        if n > n_stream_length:
            break
    plt.figure()
    plt.plot(results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tikzplotlib.save("figures/" + str(run_id) + "_oh_rolling_accuracy_arf.tex")
    plt.show()


# Run Experiment
n_runs = 100
model_parameter = pd.DataFrame(index=[0])
n_stream_length = 2000
drift_position = int(1250)
drift_width = 50
len_explanation_interval = max(300, int(n_stream_length / 20))
adwin_delta = 0.025
tau = 0.3

experiment_results = pd.DataFrame()
for run_id in range(n_runs):
    print(run_id,"/",n_runs)
    run_results = run_experiment(run_id, n_stream_length, drift_position, drift_width, len_explanation_interval,
                                 adwin_delta, tau)
    experiment_results = pd.concat([experiment_results, run_results])

experiment_means = experiment_results.groupby("Model Name").mean()
experiment_stds = experiment_results.groupby("Model Name").std()
# Output results and parameter
model_parameter["Stream Length"] = n_stream_length
model_parameter["Drift Position"] = drift_position
model_parameter["Explanation Collection Interval"] = len_explanation_interval
model_parameter["ADWIN Delta"] = adwin_delta
model_parameter["Experiment Iterations"] = n_runs
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_results.to_csv("results/" + str(run_id) + "_experiment_stagger_results.csv")
experiment_means.to_csv("results/" + str(run_id) + "_experiment_stagger_results_mean.csv")
experiment_stds.to_csv("results/" + str(run_id) + "_experiment_stagger_results_std.csv")
model_parameter.to_csv("results/" + str(run_id) + "_experiment_stagger_params.csv")

# Plot rolling accuracy for ARF
plot_accuracy(n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau)


