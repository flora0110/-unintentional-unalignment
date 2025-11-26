import argparse

from common.experiment.experiments_plan_runner import ExperimentsPlanRunner
from goodreads_experiments.experiments.goodreads_multi_example_experiment import GoodreadsMultiExampleExperiment


def main():
    parser = argparse.ArgumentParser()
    ExperimentsPlanRunner.add_experiments_plan_runner_specific_args(parser)
    args = parser.parse_args()

    experiments_plan_runner = ExperimentsPlanRunner()
    experiment = GoodreadsMultiExampleExperiment()
    experiments_plan_runner.run(plan_config_path=args.plan_config_path,
                                experiment=experiment,
                                disable_console_log=args.disable_console_log,
                                save_logs=args.save_logs,
                                log_dir=args.log_dir,
                                log_file_name_prefix=args.log_file_name_prefix)


if __name__ == "__main__":
    main()
