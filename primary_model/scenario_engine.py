"""
Built to investigate TFT forecasting when input variables change during different scenarios
"""
import torch
@torch.no_grad()
def scenario_prediction(model, encoder_df, decoder_df, feature_mns, feature_stds,
                        past_feature_names, future_feature_names, device):
    model.eval()
    past_dictionary = {
        name: torch.tensor(
            # using z score normalization that is used before training
            ((encoder_df[name] - feature_mns[name]) / feature_stds[name]).values,
            dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        for name in past_feature_names
    }

    future_dictionary = {
        name: torch.tensor(
            # using z score normalization that is used before training
            ((decoder_df[name] - feature_mns[name]) / feature_stds[name]).values,
            dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        for name in future_feature_names
    }
    forecast_n, interp = model(past_dictionary, future_dictionary)
    # reversing normalization
    forecast_mw = forecast_n * feature_stds["ontario_demand_mw"] + feature_mns["ontario_demand_mw"]
    return forecast_mw.cpu(), interp

def bandwidth(prediction):
    # quantile 0.9 - 0.1
    return (prediction[...,2] - prediction[...,0]).mean().item()

class ScenarioEngine:
    def __init__(self, model, feature_mns, feature_stds, past_feature_names,
                 future_feature_names, device):
        self.model = model
        self.feature_means = feature_mns
        self.feature_stds = feature_stds
        self.past_feature_names = past_feature_names
        self.future_feature_names = future_feature_names
        self.device = device

    def predict(self, encoder_df, decoder_df): # crucial for the forecast horizon
        # same shape as quantile output layer
        return scenario_prediction(
            self.model, encoder_df, decoder_df, self.feature_means, self.feature_stds,
            self.past_feature_names, self.future_feature_names, self.device
        )
    def apply_modifications(self, decoder_df, modifications):
        df = decoder_df.copy()
        for feat, factor in modifications.items():
            df[feat] *= factor
        return df
    def extrapolation_guard(self, testing_df, exploration_df):
        # ensures that the training data bounds the possible values in the exp df
        warning = []
        for feat in self.future_feature_names:
            if feat in exploration_df.columns:
                lower, upper = testing_df[feat].min(), testing_df[feat].max()
                e_lower, e_upper= exploration_df[feat].min(), exploration_df[feat].max()
                if e_lower < lower or e_upper > upper:
                    warning.append(f"Beware --> {feat} scenario range {e_lower}, {e_upper}"
                                   f"surpasses training range {lower}, {upper}")
    def compare(self, encoder_df, decoder_df, modifications, testing_df, scenario_name="unique"):
        # need to extract the unmodified prediction for comparisons
        base, _ = self.predict(encoder_df, decoder_df)
        scenario_df = self.apply_modifications(decoder_df, modifications)
        scenario, _ = self.predict(encoder_df, scenario_df)
        beware = self.extrapolation_guard(testing_df, scenario_df)
        # quantile = 0.5 (median)
        med_base = base[...,1]
        med_scenario = scenario[...,1]

        return {
            # extract insights to put into chart which inform planning insights for grid operators
            "modifications": modifications,
            "modification_warning": beware,
            "name_of_scenario": scenario_name,
            "base_peak_mw": med_base.max().item(),
            "scenario_peak_mw": med_scenario.max().item(),
            "peak_shift_mw": (med_scenario.max() - med_base.max()).item(),
            # hours begin at 0
            "peak_hour_base": med_base.argmax().item() + 1,
            "peak_hour_scenario": med_scenario.argmax().item() + 1,
            "average_shift_mw": (med_scenario - med_base).mean().item(),
            "hourly_base": base,
            "hourly_scenario": scenario,
            "bandwidth_base": bandwidth(base),
            "bandwidth_scenario": bandwidth(scenario)
        }

def extract_windows(df, forecast_index, past_feature_names, future_feature_names, encoder_len=168, decoder_len=24):
    past_extract = df.iloc[forecast_index - encoder_len: forecast_index]
    future_extract = df.iloc[forecast_index: forecast_index + decoder_len]
    encoder_df = past_extract[past_feature_names].copy()
    decoder_df = future_extract[future_feature_names].copy()
    return encoder_df, decoder_df