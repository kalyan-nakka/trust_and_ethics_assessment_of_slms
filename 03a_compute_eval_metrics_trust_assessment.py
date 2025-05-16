import argparse

from importlib import import_module

PERSPECTIVES = {
    "stereotype": "perspectives.stereotype.agreement_func",
    "fairness": "perspectives.fairness.score_calculation_script",
    "privacy": "perspectives.privacy.result_agg",
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

    parser.add_argument("--env",
                        type=str,
                        default="on_server",
                        choices=["on_server",
                                 "on_device"],
                        help="Name of Environment")

    return parser.parse_args()


def main():
    args = parse_arguments()
    module_name = PERSPECTIVES.get(args.p)

    perspective_module = import_module(module_name)
    perspective_module.results_eval(env=args.env)


if __name__ == '__main__':
    main()
