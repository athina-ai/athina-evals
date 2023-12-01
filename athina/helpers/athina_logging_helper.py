import time
from typing import List
from athina.interfaces.athina import (
    AthinaEvalRequestCreateRequest,
    AthinaEvalRequestSource,
    AthinaEvalResult,
    AthinaJobType,
    AthinaEvalRunResult,
    AthinaInterfaceHelper,
)
from athina.interfaces.result import LlmEvalResult
from athina.services.athina_api_service import AthinaApiService


class AthinaLoggingHelper:
    def create_eval_request(eval_name: str, request_data: dict, request_type: str):
        try:
            # Create eval request
            eval_request = AthinaEvalRequestCreateRequest(
                request_label=eval_name + "_eval_" + str(time.time()),
                request_data=request_data,
                request_data_type=request_type,
                source=AthinaEvalRequestSource.DEV_SDK.value,
            )
            eval_request_id = AthinaApiService.create_eval_request(eval_request)[
                "data"
            ]["eval_request"]["id"]
            return eval_request_id
        except Exception as e:
            print(
                f"An error occurred while creating eval request",
                str(e),
            )
            raise

    def log_eval_results(
        eval_request_id: str,
        eval_results: LlmEvalResult,
        data: List[dict],
    ):
        try:
            athina_eval_result_create_many_request = []

            for eval_result in eval_results:
                # Construct eval result object
                failed_percent = 1.0 if eval_result["failure"] else 0.0
                athina_eval_result = AthinaEvalResult(
                    job_type=AthinaJobType.LLM_EVAL.value,
                    failed_percent=failed_percent,
                    number_of_runs=1,
                    flakiness=0.0,
                    run_results=[
                        AthinaEvalRunResult(
                            failed=eval_result["failure"],
                            runtime=eval_result["runtime"],
                            reason=eval_result["reason"],
                            data=data,
                        )
                    ],
                    runtime=eval_result["runtime"],
                )

                # log eval results to Athina
                athina_eval_result_create_request = (
                    AthinaInterfaceHelper.eval_result_to_create_request(
                        eval_request_id=eval_request_id,
                        eval_type=eval_result["name"],
                        language_model_id=eval_result["model"],
                        eval_result=athina_eval_result,
                    )
                )
                athina_eval_result_create_request_dict = {
                    k: v
                    for k, v in athina_eval_result_create_request.items()
                    if v is not None
                }
                athina_eval_result_create_many_request.append(
                    athina_eval_result_create_request_dict
                )
            AthinaApiService.log_eval_results(athina_eval_result_create_many_request)

        except Exception as e:
            print(
                f"An error occurred while posting eval results",
                str(e),
            )
            raise
