import pandas as pd
from dataclasses import dataclass
from typing import TypedDict, List, Optional
from athina.interfaces.data import DataPoint


class LlmEvalResultMetric(TypedDict):
    """
    Represents the LLM evaluation result metric.
    """

    id: str
    value: float


class LlmEvalResult(TypedDict):
    """
    Represents the LLM evaluation result.
    """

    name: str
    display_name: str
    data: DataPoint
    failure: int
    reason: str
    runtime: int
    model: str
    metric: Optional[LlmEvalResultMetric]

@dataclass
class BatchRunResult:
    """
    Represents the result of a batch run of LLM evaluation.
    """

    eval_request_id: str
    eval_results: List[LlmEvalResult]

    def to_df(self):
        """
        Converts the batch run result to a Pandas DataFrame.
        """
        pd.set_option('display.max_colwidth', 500)
        df = pd.DataFrame(self.eval_results)

        # Normalize the 'data' column
        results_df = df.drop(columns=['data', 'name', 'metric']).rename(columns={'failure': 'failed', 'name': 'eval','reason': 'grade_reason'})
        data_normalized = pd.json_normalize(df['data'])
        metric_normalized = pd.json_normalize(df['metric']).rename(columns={'id': 'metric_id', 'value': 'metric_value'})

        # Concatenate the normalized data with the original DataFrame (excluding the 'data' column)
        df = pd.concat([data_normalized, results_df, metric_normalized], axis=1)
        return df


class EvalPerformanceReport(TypedDict):
    """
    Represents the performance metrics for an evaluation.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    runtime: int
    dataset_size: int
