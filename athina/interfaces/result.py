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
    metrics: List[LlmEvalResultMetric]

@dataclass
class BatchRunResult:
    """
    Represents the result of a batch run of LLM evaluation.
    """

    eval_request_id: str
    eval_results: List[LlmEvalResult]

    def to_df(self):
        """
        Converts the batch run result to a Pandas DataFrame, including data and dynamic metrics.
        """
        pd.set_option('display.max_colwidth', 500)

        df_data = []
        for item in self.eval_results:
            # Start with dynamic fields from the 'data' dictionary
            entry = {key: value for key, value in item['data'].items()}

            # Add fixed fields
            entry.update({
                'display_name': item['display_name'],
                'failed': item['failure'],
                'grade_reason': item['reason'],
                'runtime': item['runtime'],
                'model': item['model']
            })

            # Add dynamic metrics
            for metric in item['metrics']:
                entry[metric['id']] = metric['value']

            df_data.append(entry)

        df = pd.DataFrame(df_data)
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
