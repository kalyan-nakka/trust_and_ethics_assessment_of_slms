from importlib import import_module

PERSPECTIVES = {
    "stereotype": "perspectives.stereotype.agreement_func",
    # "toxicity": "perspectives.toxicity.perspective_api_evaluate",
    # "fairness": "perspectives.fairness.score_calculation_script",
    # "privacy": "perspectives.privacy.result_agg",
}


def run():

    for name, module_name in PERSPECTIVES.items():
        perspective_module = import_module(module_name)
        perspective_module.results_eval()


if __name__ == '__main__':
    run()
