from .run import EvalRunner


def run(evals, data=None, max_parallel_evals=5, dataset_id=None, number_of_rows=None):
    """
    A convenience wrapper to run evaluation suites.

    :param evals: A list of evaluations to be run.
    :param data: The dataset over which evaluations are run.
    """
    # Call the EvalRunner's run_suite method directly
    return EvalRunner.run_suite(
        evals=evals,
        data=data,
        max_parallel_evals=max_parallel_evals,
        dataset_id=dataset_id,
        number_of_rows=number_of_rows,
    )
