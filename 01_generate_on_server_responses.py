import argparse

from importlib import import_module
from configs.configs import BaseConfig, build_config
from huggingface_hub import login


PERSPECTIVES = {
    "ethical_safeguards": "perspectives.ethical_safeguards.get_responses",
    "stereotype": "perspectives.stereotype.bias_generation",
    "fairness": "perspectives.fairness.fairness_evaluation",
    "privacy": "perspectives.privacy.privacy_evaluation",
}


PERSPECTIVES_CONFIG = {
    "ethical_safeguards": False,
    "stereotype": False,
    "fairness": False,
    "privacy": False,
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--p",
                        type=str,
                        default="stereotype",
                        choices=["stereotype",
                                 "fairness",
                                 "privacy",
                                 "ethical_safeguards"],
                        help="Name of Evaluation perspective")

    return parser.parse_args()


def run(base_config: BaseConfig) -> None:
    assert isinstance(base_config, BaseConfig)
    print(base_config)

    for name, module_name in PERSPECTIVES.items():
        if getattr(base_config, name) is not None:
            perspective_module = import_module(module_name)
            perspective_module.main(base_config)


def main():
    login("huggingface_token")  # Add your Huggingface token here

    args = parse_arguments()

    PERSPECTIVES_CONFIG[args.p] = True
    config = build_config(perspectives=PERSPECTIVES_CONFIG)

    run(base_config=config)


if __name__ == "__main__":
    main()
