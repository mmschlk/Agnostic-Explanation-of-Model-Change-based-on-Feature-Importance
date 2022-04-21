# -*- coding: utf-8 -*-
from changeExplainer import changeExplainer
from river import metrics, synth, ensemble, linear_model, compose, neighbors, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tikzplotlib


def run_experiment(run_id, n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau):
    seed_value = np.random.randint(1000)
    stream_1 = synth.Agrawal(1, seed=seed_value,balance_classes=True)
    stream_2 = synth.Agrawal(2, seed=seed_value,balance_classes=True)
    stream = synth.ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        position=drift_position,
        width=drift_width
    )
    scaler = preprocessing.StandardScaler()
    oh = compose.Select('elevel','car','zipcode') | preprocessing.OneHotEncoder()
    oh += compose.Select('salary','commission','age','hvalue','hyears','loan')
    feature_list = ['salary','commission','age','elevel','car','zipcode','hvalue','hyears','loan']

    arf_accuracy = metrics.Accuracy()
    lm_accuracy = metrics.Accuracy()
    ibl_accuracy = metrics.Accuracy()
    arf_model = ensemble.AdaptiveRandomForestClassifier(seed=seed_value)
    arf_explainer = changeExplainer(arf_model, "Adaptive Random Forest", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)

    ibl_model = neighbors.SAMKNNClassifier()
    ibl_explainer = changeExplainer(ibl_model, "SAMKNN Classifier", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)



    for (n, (x, y)) in enumerate(stream):
        scaler.learn_one(x)
        x=scaler.transform_one(x)
        # Prediction
        y_pred_arf = arf_model.predict_one(x)
        y_pred_ibl = ibl_model.predict_one(x)
        # Accuracy Update
        arf_accuracy.update(y, y_pred_arf)
        ibl_accuracy.update(y,y_pred_ibl)
        # Learning
        arf_model.learn_one(x, y)
        ibl_model.learn_one(x,y)
        # Explaining
        arf_explainer.explain_sample(x, y, y_pred_arf)
        ibl_explainer.explain_sample(x,y,y_pred_ibl)
        arf_explainer.explanations["Total Accuracy"] = arf_accuracy.get()
        ibl_explainer.explanations["Total Accuracy"] = ibl_accuracy.get()
        if n > n_stream_length:
            break

    results = pd.concat([arf_explainer.explanations, ibl_explainer.explanations])
    results["run_id"] = run_id
    return results


def plot_accuracy(n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau,
                  seed_value=43, window_size=200):
    results = []
    stream_1 = synth.Agrawal(1, balance_classes=True,seed=seed_value)
    stream_2 = synth.Agrawal(2, balance_classes=True,seed=seed_value)
    stream = synth.ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        position=drift_position,
        width=drift_width,
        seed=seed_value
    )
    scaler = preprocessing.StandardScaler()
    oh = compose.Select('elevel','car','zipcode') | preprocessing.OneHotEncoder()
    oh += compose.Select('salary','commission','age','hvalue','hyears','loan')
    feature_list = ['salary','commission','age','elevel','car','zipcode','hvalue','hyears','loan']
    arf_rolling_accuracy = metrics.Rolling(metrics.Accuracy(), window_size=window_size)
    arf_model = ensemble.AdaptiveRandomForestClassifier(seed=seed_value)
    arf_explainer = changeExplainer(arf_model, "Adaptive Random Forest", len_explanation_interval, adwin_delta, tau,
                                    seed_value,feature_list)


    for (n, (x, y)) in enumerate(stream):

        scaler.learn_one(x)
        x=scaler.transform_one(x)
        # Prediction
        y_pred_arf = arf_model.predict_one(x)
        # Accuracy Update
        arf_rolling_accuracy.update(y, y_pred_arf)
        if n % 50 == 0:
            results.append(arf_rolling_accuracy.get())
        # Learning
        arf_model.learn_one(x, y)
        # Explaining
        arf_explainer.explain_sample(x, y, y_pred_arf)
        if n > n_stream_length:
            break
    plt.figure()
    plt.plot(range(0,len(results)*50,50),results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tikzplotlib.save("figures/"+str(run_id)+"_agrawal_rolling_accuracy_arf.tex")
    plt.show()

# Run Experiment
n_runs = 50
model_parameter = pd.DataFrame(index=[0])
n_stream_length = 20000
drift_position = int(n_stream_length*2/3)
drift_width = 50
len_explanation_interval = max(300, int(n_stream_length / 20))
adwin_delta = 0.025
tau = 0.4

experiment_results = pd.DataFrame()
for run_id in range(n_runs):
    print(run_id,"/",n_runs," : ",datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_results = run_experiment(run_id, n_stream_length, drift_position, drift_width, len_explanation_interval,
                                 adwin_delta, tau)
    experiment_results = pd.concat([experiment_results, run_results])

# Plot rolling accuracy for ARF
plot_accuracy(n_stream_length, drift_position, drift_width, len_explanation_interval, adwin_delta, tau)

experiment_means = experiment_results.groupby("Model Name").mean()
experiment_stds = experiment_results.groupby("Model Name").std()
#Output results and parameter
model_parameter["Stream Length"] = n_stream_length
model_parameter["Drift Position"] = drift_position
model_parameter["Explanation Collection Interval"] = len_explanation_interval
model_parameter["ADWIN Delta"] = adwin_delta
model_parameter["Experiment Iterations"] = n_runs
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_results.to_csv("results/"+str(run_id)+"_experiment_agrawal_results.csv")
experiment_means.to_csv("results/"+str(run_id)+"_experiment_agrawal_results_mean.csv")
experiment_stds.to_csv("results/"+str(run_id)+"_experiment_agrawal_results_std.csv")
model_parameter.to_csv("results/"+str(run_id)+"_experiment_agrawal_params.csv")


means = experiment_results.groupby("Model Name").mean()
deltas = means.loc[:,means.columns.str.startswith('PFI_delta')].transpose()
plt.figure()
plt.bar(deltas.index.str.slice(10),deltas.iloc[:,0],label="ARF")
plt.legend()
plt.show()

plt.figure()
plt.bar(deltas.index.str.slice(10),deltas.iloc[:,1],label="SAM-kNN")
plt.legend()
plt.show()