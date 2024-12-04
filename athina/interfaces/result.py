import pandas as pd
from dataclasses import dataclass, field
from typing import TypedDict, List, Optional
from athina.interfaces.data import DataPoint
from pydantic import BaseModel


class EvalResultMetric(TypedDict):
    """
    Represents the LLM evaluation result metric.
    """

    id: str
    value: float


class DatapointFieldAnnotation(TypedDict):
    """
    The annotations to be logged for the datapoint field.
    """

    field_name: str
    text: str
    annotation_type: str
    annotation_note: str


class EvalResult(TypedDict):
    """
    Represents the LLM evaluation result.
    """

    name: str
    display_name: str
    data: dict
    failure: Optional[bool]
    reason: str
    runtime: int
    model: Optional[str]
    metrics: List[EvalResultMetric]
    datapoint_field_annotations: Optional[List[DatapointFieldAnnotation]]
    metadata: Optional[dict]


@dataclass
class BatchRunResult:
    """
    Represents the result of a batch run of LLM evaluation.
    """

    eval_results: List[Optional[EvalResult]]
    eval_request_id: Optional[str] = field(default=None)

    def to_df(self):
        """
        Converts the batch run result to a Pandas DataFrame, including data and dynamic metrics.
        """
        pd.set_option("display.max_colwidth", 500)

        df_data = []
        for item in self.eval_results:
            if item is None:
                # Add a representation for None entries
                entry = {
                    "display_name": None,
                    "failed": None,
                    "grade_reason": None,
                    "runtime": None,
                    "model": None,
                    # Add more fields as None or with a placeholder as necessary
                }
            else:
                # Start with dynamic fields from the 'data' dictionary
                entry = {key: value for key, value in item["data"].items()}

                # Add fixed fields
                entry.update(
                    {
                        "display_name": item["display_name"],
                        "failed": item.get("failure"),
                        "grade_reason": item["reason"],
                        "runtime": item["runtime"],
                        "model": item.get("model"),
                    }
                )

                # Add dynamic metrics
                for metric in item["metrics"]:
                    entry[metric["id"]] = metric["value"]

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


class GuardResult(BaseModel):
    passed: bool
    reason: str
    runtime: int
