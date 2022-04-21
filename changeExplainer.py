from river import drift
import pandas as pd
from copy import deepcopy
import numpy as np

class changeExplainer:
    def __init__(self,model,model_name,len_explanation_interval,adwin_delta,tau,seed_value,feature_list,print=False):
        self.model = model
        self.model_name = model_name
        self.len_explanation_interval = len_explanation_interval
        self.adwin = drift.ADWIN(delta=adwin_delta)
        self.tau = tau
        self.seed_value = seed_value
        self.feature_list = feature_list
        self.print = print
        #Initialize remaining
        self.timestep_s = 0
        self.remaining_samples = 0
        self.feature_batch_s = pd.DataFrame()
        self.target_batch_s = pd.DataFrame()
        self.feature_batch_t = pd.DataFrame()
        self.target_batch_t = pd.DataFrame()
        self.is_ref_set_init = False
        self.is_ref_set_finished = False
        self.in_drift = False
        self.explanations = pd.DataFrame()
        self.drift_detections = np.array([])
        self.model_history = []
        self.n = 0
        self.PFI_s = pd.DataFrame(index=[0])

    def calc_expected_discrepancy(self,samples, model_s, model_t):
        rslt = 0.0
        for i, sample in samples.iterrows():
            sample_dict = sample.to_dict()
            rslt += (model_s.predict_one(sample_dict) - model_t.predict_one(sample_dict)) ** 2
        return rslt / len(samples)
    def calc_pfi(self,feature_batch_pfi, target_batch_pfi, model, feature_name):
        permutation = np.random.permutation(feature_batch_pfi.index)
        permuted_feature_values = pd.DataFrame(feature_batch_pfi.loc[permutation, feature_name].to_numpy(),
                                               index=feature_batch_pfi.index,columns=feature_name)
        permuted_feature_batch = deepcopy(feature_batch_pfi)
        permuted_feature_batch.loc[:, feature_name] = permuted_feature_values
        acc_batch = 0
        acc_perm = 0
        for i, obs in feature_batch_pfi.iterrows():
            acc_batch += model.predict_one(obs.to_dict()) != target_batch_pfi.loc[i]
        for i, obs in permuted_feature_batch.iterrows():
            acc_perm += model.predict_one(obs.to_dict()) != target_batch_pfi.loc[i]
        return (acc_perm-acc_batch) / len(feature_batch_pfi)
    def explain_sample(self,x,y,y_pred):
        if self.is_ref_set_finished:
            self.in_drift, self.in_warning = self.adwin.update(1 if y_pred == y else 0)
        # Initialize reference model
        if not self.is_ref_set_init:
            self.remaining_samples = self.len_explanation_interval
            self.is_ref_set_init = True
        # Start collection when drift occurred
        if self.in_drift and self.remaining_samples==0:
            if self.print:
                print("Start collection")
            self.remaining_samples = self.len_explanation_interval
            self.timestep_t = self.n+self.len_explanation_interval
            self.drift_detections=np.append(self.drift_detections,self.n)
        # Collect data
        if self.remaining_samples > 0:
            if self.is_ref_set_finished==False:
                # store sample in reference
                self.feature_batch_s = pd.concat([self.feature_batch_s, pd.DataFrame(x, index=[self.n])])
                self.target_batch_s = pd.concat([self.target_batch_s, pd.DataFrame([y], index=[self.n])])
            else:
                # store sample in actual
                self.feature_batch_t = pd.concat([self.feature_batch_t, pd.DataFrame(x, index=[self.n])])
                self.target_batch_t = pd.concat([self.target_batch_t, pd.DataFrame([y], index=[self.n])])
            if self.remaining_samples == 1:
                # at the end of collection store model
                if self.is_ref_set_finished:
                    expected_discrepancy = self.calc_expected_discrepancy(self.feature_batch_s.append(self.feature_batch_t),self.model_history[-1], self.model)
                    if expected_discrepancy > self.tau:
                        # If semantic model change is significant, create explanation
                        if self.print:
                            print("Creating Explanations at observation: ", self.n)
                        # Explain model difference
                        explanation = pd.DataFrame()
                        for feature_id in self.feature_list:
                            feature = self.feature_batch_s.columns[self.feature_batch_s.columns.str.startswith(feature_id)]
                            explanation["PFI_s_" + feature_id] = self.PFI_s[feature_id]
                            explanation["PFI_t_" + feature_id] = self.calc_pfi(self.feature_batch_t, self.target_batch_t, self.model, feature)
                            explanation["PFI_delta_" + feature_id] = explanation["PFI_t_" + feature_id] - explanation["PFI_s_" + feature_id]
                            #Reset reference
                            self.PFI_s[feature_id] = explanation["PFI_t_" + feature_id]
                        explanation["Expected Discrepancy"] = expected_discrepancy
                        explanation["Time Step s"] = self.timestep_s
                        explanation["Time Step t"] = self.timestep_t
                        explanation["Model Name"] = self.model_name
                        explanation["Drift detected at"] = self.drift_detections[-1]
                        self.explanations = pd.concat([self.explanations,explanation])
                        self.feature_batch_s = deepcopy(self.feature_batch_t)
                        self.target_batch_s = deepcopy(self.target_batch_t)
                        self.timestep_s = self.n + 1
                        self.model_history.append(deepcopy((self.model)))
                    else:
                        if self.print:
                            print("Expected Discrepancy not exceeded")
                    # either way, empty batches
                    self.feature_batch_t = pd.DataFrame()
                    self.target_batch_t = pd.DataFrame()
                else:
                    #Compute initial PFI
                    for feature_id in self.feature_list:
                        feature = self.feature_batch_s.columns[self.feature_batch_s.columns.str.startswith(feature_id)]
                        self.PFI_s[feature_id] = self.calc_pfi(self.feature_batch_s, self.target_batch_s,self.model, feature)
                    self.is_ref_set_finished = True
                    self.timestep_s = self.n + 1
                    self.model_history.append(deepcopy((self.model)))
            self.remaining_samples -= 1
        self.n += 1
        return self.explanations
