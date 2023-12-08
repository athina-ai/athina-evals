import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, TypedDict, Any
from .openai import OpenAiPromptMessage


@dataclass
class AthinaInference:
    """Athina PromptRun class"""

    id: str
    prompt_slug: Optional[str]
    language_model_id: Optional[str]
    user_query: Optional[str]
    context: Optional[Dict[str, str]]
    prompt_response: Optional[str]


@dataclass
class AthinaFilters:
    prompt_slug: Optional[str] = None
    language_model_id: Optional[str] = None
    environment: Optional[str] = None
    topic: Optional[str] = None
    customer_id: Optional[str] = None

    def to_dict(self) -> str:
        return asdict(self)


class AthinaEvalRunResult(TypedDict):
    failed: bool
    runtime: float
    reason: str


class AthinaEvalResult(TypedDict):
    job_type: str
    failed_percent: Optional[float]
    number_of_runs: int
    flakiness: float
    run_results: List[AthinaEvalRunResult]
    runtime: float
    data: Dict


class AthinaEvalRequestSource(Enum):
    DEV_SDK = "dev_sdk"
    SCHEDULED_JOB = "scheduled_job"
    UI_DASHBOARD = "ui_dashboard"


class AthinaEvalRequestCreateRequest(TypedDict):
    request_label: str
    request_data: Dict[str, Any]
    request_data_type: str
    source: str


class AthinaEvalResultCreateRequest(TypedDict):
    org_id: Optional[str]
    prompt_run_id: Optional[str]
    job_config_id: Optional[str]
    eval_job_id: Optional[str]
    language_model_id: Optional[str]
    job_type: str
    eval_type_id: str
    run_results: List[AthinaEvalRunResult]
    data: Dict
    eval_request_id: Optional[str]
    number_of_runs: int
    flakiness: float
    runtime: int
    failed_percent: Optional[float]


class AthinaJobType(Enum):
    LLM_EVAL = "LlmEval"


class AthinaInterfaceHelper:
    @staticmethod
    def eval_result_to_create_request(
        eval_request_id: str,
        eval_type: str,
        language_model_id: str,
        eval_result: AthinaEvalResult,
    ) -> AthinaEvalResultCreateRequest:
        return AthinaEvalResultCreateRequest(
            org_id=None,
            prompt_run_id=None,
            job_config_id=None,
            eval_job_id=None,
            language_model_id=language_model_id,
            job_type=eval_result["job_type"],
            eval_type_id=eval_type,
            failures=[],
            eval_result=eval_result,
            prompt_run_updates={},
            run_results=eval_result["run_results"],
            data=eval_result["data"],
            eval_request_id=eval_request_id,
            number_of_runs=eval_result["number_of_runs"],
            flakiness=eval_result["flakiness"],
            runtime=eval_result["runtime"],
            failed_percent=eval_result["failed_percent"],
        )


class AthinaExperiment(TypedDict):
    experiment_name: str
    experiment_description: str
    language_model_provider: str
    language_model_id: str
    prompt_template: List[OpenAiPromptMessage]
    dataset_name: str
