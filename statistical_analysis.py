import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (wilcoxon,
                         rankdata)


stereotype_results_dirs = {
    "on_server": "./results/stereotype/generations/",
    "on_device": "./results_mobile/stereotype/"
}


privacy_results_dirs = {
    "on_server": "results/privacy/generations/pii",
    "on_device": "results_mobile/privacy/generations/pii"
}


slms_names_map = {
    "gemma": "Gemma-2B",
    "phi": "Phi-2",
    "RedPajama": "RedPajama-3B"
}


env_names_map = {
    "on_server": "On-Server",
    "on_device": "On-Device"
}


def effect_size(before, after):
    # Calculate differences
    differences = after - before

    # Rank the differences
    ranks = rankdata(np.abs(differences))

    # Signed ranks
    signed_ranks = np.sign(differences) * ranks

    # Calculate W (sum of positive ranks)
    W = np.sum(signed_ranks[signed_ranks > 0])

    # Calculate the mean and standard deviation of W
    n = len(differences)
    mean_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    # Calculate z-score
    z = (W - mean_W) / std_W

    # Calculate effect size r
    r = z / np.sqrt(n)

    return r


def stereotype_analysis(models):
    a_i_data = []
    print("\nStereotype Statistical Analysis:\n")
    for model in models:
        statistical_data = {}
        for environment in stereotype_results_dirs.keys():
            dir_path = stereotype_results_dirs.get(environment)
            all_agreement_indices = []
            for system_prompt_type in ['benign', 'untargeted', 'targeted']:
                csv_file_path = os.path.join(dir_path, model, system_prompt_type)
                agreement_df = pd.read_csv(f"{csv_file_path}/25_agreement_matrix.csv", index_col=0)
                agreement_indices = agreement_df.values.flatten()
                all_agreement_indices.extend(agreement_indices)
            statistical_data[environment] = all_agreement_indices

        on_server = np.array(statistical_data.get("on_server"))
        on_device = np.array(statistical_data.get("on_device"))

        stat, p_value = wilcoxon(on_server, on_device)
        eff_size = effect_size(before=on_server, after=on_device)
        print(f'{model} WSRT Results\nStatistic: {stat}, p-value: {p_value}, effect size: {eff_size}\n')

        for environment in stereotype_results_dirs.keys():
            for a_i in statistical_data.get(environment):
                a_i_data.append({
                    "Small Language Model(s)": slms_names_map.get(model),
                    "Environment": env_names_map.get(environment),
                    "Model Agreement": a_i
                })

    a_i_data_df = pd.DataFrame(a_i_data)
    # Box Plot
    plt.figure(figsize=(50, 40))
    plt.rcParams.update({'font.size': 90})
    sns.boxplot(data=a_i_data_df,
                x="Small Language Model(s)",
                y="Model Agreement",
                hue="Environment",
                fill=False,
                linewidth=20,
                gap=0.1)
    plt.tick_params(axis='x', labelsize=100)
    plt.tick_params(axis='y', labelsize=100)

    plt.legend(fontsize=100)
    plt.ylabel(r"Model Stereotype Agreement $(\it{A_{i}})$", fontweight="bold", fontsize=100)
    plt.xlabel("")
    plt.title(r"Distribution of $\it{A_{i}}$ per each SLM", fontweight="bold", fontsize=100)
    plt.savefig(f"dist_stereotype_scores.jpg")


def fairness_analysis(models):
    m_dpd_results = {
        "gemma": {
            "on_server": [0.02, 0.02, 0.07, 0.11, 0.05, 0.05],
            "on_device": [0.04, 0.05, 0.07, 0.13, 0.08, 0.06],
        },
        "phi": {
            "on_server": [0.00, 0.02, 0.04, 0.06, 0.01, 0.06],
            "on_device": [0.04, 0.02, 0.06, 0.17, 0.20, 0.17],
        },
        "RedPajama": {
            "on_server": [0.00, 0.00, 0.00, 0.08, 0.08, 0.02],
            "on_device": [0.02, 0.02, 0.06, 0.14, 0.11, 0.16],
        },
    }
    m_dpd_data = []
    print("\nFairness Statistical Analysis:\n")
    for model in models:
        statistical_data = m_dpd_results.get(model)

        on_server = np.array(statistical_data.get("on_server"))
        on_device = np.array(statistical_data.get("on_device"))

        stat, p_value = wilcoxon(on_server, on_device)
        eff_size = effect_size(before=on_server, after=on_device)
        print(f'{model} WSRT Results\nStatistic: {stat}, p-value: {p_value}, effect size: {eff_size}\n')

        for environment in stereotype_results_dirs.keys():
            for m_dpd in statistical_data.get(environment):
                m_dpd_data.append({
                    "Small Language Model(s)": slms_names_map.get(model),
                    "Environment": env_names_map.get(environment),
                    "Model Unfairness": m_dpd
                })

    m_dpd_data_df = pd.DataFrame(m_dpd_data)
    # Box Plot
    plt.figure(figsize=(50, 40))
    plt.rcParams.update({'font.size': 90})
    sns.boxplot(data=m_dpd_data_df,
                x="Small Language Model(s)",
                y="Model Unfairness",
                hue="Environment",
                palette="flare",
                fill=False,
                linewidth=20,
                gap=0.1)
    plt.tick_params(axis='x', labelsize=100)
    plt.tick_params(axis='y', labelsize=100)

    plt.legend(fontsize=100)
    plt.ylabel(r"Model Unfairness $(\it{M_{dpd}})$", fontweight="bold", fontsize=100)
    plt.xlabel("")
    plt.title(r"Distribution of $\it{M_{dpd}}$ per each SLM", fontweight="bold", fontsize=100)
    plt.savefig(f"dist_fairness_scores.jpg")


def privacy_analysis(models):
    lr_data = []
    print("\nPrivacy Statistical Analysis:\n")
    for model in models:
        statistical_data = {}
        for environment in privacy_results_dirs.keys():
            all_leakage_rates = []
            dir_path = privacy_results_dirs.get(environment)

            agreement_df = pd.read_csv(f"{dir_path}/pii_{model}_leakage_rate_matrix.csv", index_col=0)
            agreement_indices = agreement_df.values.flatten()
            all_leakage_rates.extend(agreement_indices)

            statistical_data[environment] = all_leakage_rates

        on_server = np.array(statistical_data.get("on_server"))
        on_device = np.array(statistical_data.get("on_device"))

        stat, p_value = wilcoxon(on_server, on_device)
        eff_size = effect_size(before=on_server, after=on_device)
        print(f'{model} WSRT Results\nStatistic: {stat}, p-value: {p_value}, effect size: {eff_size}\n')

        for environment in privacy_results_dirs.keys():
            for lr in statistical_data.get(environment):
                lr_data.append({
                    "Small Language Model(s)": slms_names_map.get(model),
                    "Environment": env_names_map.get(environment),
                    "Model Leakage Rate": lr
                })

    lr_data_df = pd.DataFrame(lr_data)
    # Box Plot
    plt.figure(figsize=(50, 40))
    plt.rcParams.update({'font.size': 90})
    sns.boxplot(data=lr_data_df,
                x="Small Language Model(s)",
                y="Model Leakage Rate",
                hue="Environment",
                palette="dark:#5A9_r",
                fill=False,
                linewidth=20,
                gap=0.1)
    plt.tick_params(axis='x', labelsize=100)
    plt.tick_params(axis='y', labelsize=100)

    plt.legend(fontsize=100)
    plt.ylabel(r"Model PII Leakage Rate $(\it{LR})$", fontweight="bold", fontsize=100)
    plt.xlabel("")
    plt.title(r"Distribution of $\it{LR}$ per each SLM", fontweight="bold", fontsize=100)
    plt.savefig(f"dist_privacy_scores.jpg")


if __name__ == '__main__':
    all_models = ["gemma", "phi", "RedPajama"]
    stereotype_analysis(models=all_models)
    fairness_analysis(models=all_models)
    privacy_analysis(models=all_models)
