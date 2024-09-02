from importlib import import_module
from configs.configs import BaseConfig, build_config
from huggingface_hub import login
# from summarize import summarize_results

PERSPECTIVES = {
    "do_not_answer": "perspectives.do_not_answer.get_responses",
    # "stereotype": "perspectives.stereotype.bias_generation",
    # "fairness": "perspectives.fairness.fairness_evaluation",
    # "privacy": "perspectives.privacy.privacy_evaluation",
    # "advglue": "perspectives.advglue.gpt_eval",                                          # NOT Interested
    # "toxicity": "perspectives.toxicity.text_generation_hydra",                           # NOT Interested
    # # "adv_demonstration": "perspectives.adv_demonstration.adv_demonstration_hydra",     # NOT Interested
    # "machine_ethics": "perspectives.machine_ethics.test_machine_ethics",                 # NOT Interested
    # # "ood": "perspectives.ood.evaluation_ood"                                           # NOT Interested
}


def run(base_config: BaseConfig) -> None:
    # The 'validator' methods will be called when you run the line below
    # config: BaseConfig = OmegaConf.to_object(config)
    assert isinstance(base_config, BaseConfig)
    print(base_config)

    for name, module_name in PERSPECTIVES.items():
        if getattr(base_config, name) is not None:
            perspective_module = import_module(module_name)
            perspective_module.main(base_config)

    # summarize_results()


if __name__ == "__main__":
    # login("huggingface_token")

    config = build_config(perspectives=PERSPECTIVES)
    run(base_config=config)
