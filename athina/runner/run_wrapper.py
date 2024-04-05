from .run import EvalRunner

def run(evals, data):
    """
    A convenience wrapper to run evaluation suites.

    :param evals: A list of evaluations to be run.
    :param data: The dataset over which evaluations are run.
    """
    # Call the EvalRunner's run_suite method directly
    return EvalRunner.run_suite(evals=evals, data=data)