import time
from typing import List, Optional
from athina.interfaces.model import Model
from athina.interfaces.result import LlmEvalResult, LlmEvalResultMetric, BatchRunResult
from athina.loaders.summary_loader import SummaryDataPoint
from athina.metrics.metric_type import MetricType
from ..llm_evaluator import LlmEvaluator
from ..eval_type import AthinaEvalTypeId
from ..example import FewShotExample
from .question_answerer import QuestionAnswerer
from .question_generator import QuestionGenerator

class SummaryAccuracy(LlmEvaluator):
    """
    This evaluator can be configured with custom examples and instructions.
    """

    def __init__(
        self,
        questions=None,
        n_questions=5,
        model="gpt-3.5-turbo",
        metrics=[MetricType.AGREEMENT_SCORE],
    ):
        """
        Initialize the evaluator with given parameters.

        Args:
        - loader: An instance of SummarizationLoader.
        - n_questions: Number of questions to generate for summaries.
        - llm_model: Language model to be used.
        - metrics: List of metrics for evaluation.
        """

        # Intialize LLMs
        self._model = model
        self.n_questions = n_questions
        self.questions_defined = None
        if questions is None:
            self.question_generator = QuestionGenerator(
                self._model, n_questions
            )
        else:
            self.question_generator = None
            self.questions_defined = questions
        self.question_answerer = QuestionAnswerer(self._model)
        self.n_instances = 0
        # Intialize metrics
        self.metrics: List[MetricType] = metrics
        self.label_counts = {}
        for metric in metrics:
            setattr(self, f"{metric}_scores", {})

    @property
    def name(self):
        return AthinaEvalTypeId.CUSTOM.value
        
    @property
    def metric_id(self) -> str:
        return MetricType.AGREEMENT_SCORE.value
        
    @property
    def display_name(self):
        return "Summary Accuracy"
        
    @property
    def default_model(self):
        return Model.GPT35_TURBO.value
        
    @property
    def required_args(self):
        return ["document", "response"]
        
    @property
    def examples(self):
        return []
        
    def is_failure(self) -> bool:
        return False
    
    def reason(self) -> str:
        return ""


    def _evaluate(self, **instance) -> LlmEvalResult:
        """
        Run the LLM evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        self._validate_args(**instance)

        summary_datapoint = SummaryDataPoint(**instance)

        # Run the Summary Accuracy evaluator
        print("Running summary eval")
        summary_eval_result = self._evaluate_element(summary_datapoint)
        print(summary_eval_result)

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        
        llm_eval_result = LlmEvalResult(
            name=self.name,
            display_name=self.display_name,
            data=SummaryDataPoint(**instance),
            failure=self.is_failure(),
            reason=self.reason(),
            runtime=eval_runtime_ms,
            model=self._model,
            metric={
                "id": self.metric_id,
                "value": summary_eval_result[self.metric_id],
            },
        )

        return {k: v for k, v in llm_eval_result.items() if v is not None}
    

    # METHODS FROM ARIADNE
    def _evaluate_element(self, instance: SummaryDataPoint):
        """Evaluate an instance for hallucination."""
        document = instance["document"]
        summary = instance["response"]
        if "label" in instance:
            label = instance["label"]
        else:
            label = "overall"

        # Generate questions based on summary
        if self.questions_defined is None:
            questions = self.question_generator.generate(summary)
        # Or load the pre-defined questions:
        else:
            questions = self.questions_defined

        # Get answers from document and summary
        answers_doc = self.question_answerer.answer(questions, document)
        answers_sum = self.question_answerer.answer(questions, summary)
        metric_results = {}
        # Compute metrics
        if answers_doc is None or answers_sum is None or questions is None:
            metric_results["evaluation"] = "undefined"
        else:
            for metric in self.metrics:
                metric_name = metric.value
                metric_class = metric.get_class()
                metric_result, explanation = metric_class.compute(
                    answers_doc, answers_sum, questions, self.n_questions
                )
                metric_results[metric_name] = metric_result
                metric_results[f"reason_{metric_name}"] = explanation
                self.update_metric_aggregated_score(metric_name, label, metric_result)
            self.n_instances = self.n_instances + 1
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        return {
            "document": document,
            "summary": summary,
            "questions": questions,
            "answers_doc": answers_doc,
            "answers_sum": answers_sum,
            "label": label,
            **metric_results,
        }

    def update_metric_aggregated_score(self, metric, label, aggr_score):
        """Update the aggregated score for a specific metric and label."""
        metric_aggregated_scores = getattr(self, f"{metric}_scores", {})
        current_score = metric_aggregated_scores.get(label, 0)
        metric_aggregated_scores[label] = current_score + aggr_score
        setattr(self, f"{metric}_scores", metric_aggregated_scores)

    def get_metric_aggr(self, metric, label):
        """Compute the average scores based on the provided score dictionary."""
        metric_aggr = getattr(self, f"{metric}_scores", {})
        return metric_aggr.get(label, None)

    def get_average_scores(self, score_dict):
        """Compute average scores for a metric"""
        avg_scores = {}
        sum_score = 0
        n_instances = 0
        for label_type, total_score in score_dict.items():
            avg_scores[label_type] = total_score / self.label_counts[label_type]
            sum_score = sum_score + total_score
            n_instances = n_instances + self.label_counts[label_type]
        avg_scores["overall"] = sum_score / n_instances
        return avg_scores

    def compute_average_scores(self):
        """Compute average scores for each metric."""
        avg_scores = {}
        for metric in self.metrics:
            scores = getattr(self, f"{metric}_scores")
            avg_score = self.get_average_scores(scores)
            avg_scores[metric] = avg_score
        return avg_scores

    