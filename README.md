[Documentation](https://docs.athina.ai/evals) | [Athina SDK Demo Video](https://www.loom.com/share/10e37f1ba11242ac8c97902edd2fa61e) | [Athina Platform Demo Video →](https://bit.ly/athina-demo-feb-2024)

## Overview

**Athina is an open-source library with plug-and-play preset evals designed to help engineers systematically improve their LLM reliability and performance through eval-driven-development.**

![develop-ui-results-metrics-5-bg](https://github.com/athina-ai/athina-evals/assets/7515552/c6dca515-f30f-4edf-965c-b6afb1721c22)


**Quick Links**
- [Documentation](https://docs.athina.ai/evals)
- [Quick Start](https://docs.athina.ai/evals/quick_start)
- [Preset Evals](https://docs.athina.ai/evals/preset_evals)
- [Run an eval](https://docs.athina.ai/evals/running_evals/run_eval)
- [Run an eval suite](https://docs.athina.ai/evals/running_evals/run_eval_suite)
- [Customize an eval](https://docs.athina.ai/evals/custom_evals)
- [View Results on Athina Dashboard](https://docs.athina.ai/evals/develop_dashboard)
- [Loading Data for Evals](https://docs.athina.ai/evals/loading_data)
- [Cookbooks](https://docs.athina.ai/evals/cookbooks)
- [Production Monitoring](https://docs.athina.ai/monitoring)

<br />

### Why you need evals

Evaluations (evals) play a crucial role in assessing the performance of LLM responses, especially when scaling from prototyping to production.

They are akin to unit tests for LLM applications, allowing developers to:

- Catch and prevent hallucinations and bad outputs
- Measure the performance of model
- Run quantifiable experiments against ambiguous, unstructured text data
- A/B test different models and prompts rapidly
- Detect regressions before they get to production
- Monitor production data with confidence

<br />

### 🔴 Problem: Flaws with Current LLM Developer Workflows

The journey from a demo AI to a reliable production application is not easy.

Developers usually start iterating on performance by manually inspecting the outputs. Eventually they progress to using spreadsheets, CSVs, or evaluating against a golden dataset.

Each method has drawbacks, requires different tooling, and evaluation methods. [See more](https://docs.athina.ai/evals/llm_dev_workflows)

A lot of manual effort is required to set up a good infrastructure for running evals - creating a dataset, reviewing the responses, creating evals, and internal tooling / dashboard, tracking experiment parameters and metrics for historical record.

Eventually every LLM developer realizes the indispensable need for evals and an infrastructure to consistently run and track iterations to improve performance and reliability systematically.

<br />

### **🟢 Solution: Athina Evals**

[Github](https://github.com/athina-ai/athina-evals) | [Watch Demo Video](https://www.loom.com/share/10e37f1ba11242ac8c97902edd2fa61e) | [Docs](https://docs.athina.ai/evals)

Athina is an open-source library that offers a system for eval-driven development, overcoming the limitations of traditional workflows.

Our solution allows for rapid experimentation, and customizable evaluators with consistent metrics.

Here’s why this is better than building in-house eval infrastructure:

* **Plug-and-Play** [**Preset Evals**](https://docs.athina.ai/evals/preset_evals): Ready-to-use evals for immediate application
* **Integrated Dashboard**: For tracking experiments and inspecting the results in a web UI.
* **Custom Evaluators** : A flexible framework to craft tailored evals.
* **Consistent Metrics**: Uniform evaluation standards across all stages. Evaluate your model in dev and prod using a consistent set of metrics.
* **Historical Record**: Automatic tracking of every prompt iteration.
* **Quick Start**: Easy 5-min set up.

Here’s a [demo video](https://www.loom.com/share/10e37f1ba11242ac8c97902edd2fa61e).

<br /><br />

## Quick Start

The easiest way to get started is to use one of our [Example Notebooks](https://docs.athina.ai/evals/cookbooks) as a starting point.

To get started with Athina Evals:

**1. Install the `athina` package**

```
pip install athina
```

<br />

**2. Set your API keys**

If you are using the python SDK, then can set the API keys like this:

```python
from athina.keys import AthinaApiKey, OpenAiApiKey

OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
```

If you are using the CLI, then run `athina init`, and enter the API keys when prompted.

<br />

**3. Load your dataset like this:**

_You can also [load data](/evals/loading_data) using a CSV or Python Dictionary_

```python
from athina.loaders import RagLoader

dataset = RagLoader().load_json(json_filepath)
```
<br />

**4. Now you can run evals like this.**

```python
from athina.evals import DoesResponseAnswerQuery

DoesResponseAnswerQuery().run_batch(data=dataset)
```

<br />

For more detailed guides, you can follow the links below to get started running evals using Athina.

- [Quick Start Guide](https://docs.athina.ai/evals/quick_start)
- [Run an eval](https://docs.athina.ai/evals/running_evals/run_eval)
- [Run an eval suite](https://docs.athina.ai/evals/running_evals/run_eval_suite)
- [Customize an eval](https://docs.athina.ai/evals/custom_evals)
- [View Results on Athina Dashboard](https://docs.athina.ai/evals/develop_dashboard)
- [Loading Data for Evals](https://docs.athina.ai/evals/loading_data)

<br />

## Preset Evals

You can use our preset evaluators to add evaluation to your dev stack rapidly.

Here are the preset evaluators in this library:

#### RAG Evals

[These evals](https://docs.athina.ai/evals/preset_evals/rag_evals) are useful for evaluating LLM applications with Retrieval Augmented Generation (RAG).

- [Context Contains Enough Information](https://docs.athina.ai/evals/preset_evals/ccei)
- [Context Relevance](https://docs.athina.ai/evals/preset_evals/context_relevance)
- [Answer Relevance](https://docs.athina.ai/evals/preset_evals/answer_relevance)
- [Does Response Answer Query](https://docs.athina.ai/evals/preset_evals/draq)
- [Response Faithfulness](https://docs.athina.ai/evals/preset_evals/faithfulness)

<br />

_We have also built other evaluators that are not yet a part of this library (but will soon be)_
_You can find more information about these in our documentation._

#### Summarization Accuracy Evals:

These evals are useful for evaluating LLM-powered summarization performance.

- [Summarization Hallucination](https://docs.athina.ai/evals/preset_evals/summarization_eval)
- [Summarization Informativeness](https://docs.athina.ai/evals/preset_evals/summarization_eval)

<br />

#### More Evals
- [Other Evals](https://docs.athina.ai/evals/preset_evals/other_evals)

<br /><br />

## Custom Evals

See this page for more information, on how to write your own [custom evals](https://docs.athina.ai/evals/custom_evals).

<br />

---

### Why should I use Athina's Evals instead of writing my own?

You could build your own eval system from scratch, but here's why Athina might be better for you:

- Athina provides you with plug-and-play preset evals that have been well-tested
- Athina evals can run on both development and production, giving you consistent metrics for evaluating model performance and drift.
- Athina removes the need for your team to write boilerplate loaders, implement LLMs, normalize data formats, etc
- Athina offers a modular, extensible framework for writing and running evals
- Athina calculate analytics like pass rate and flakiness, and allows you to batch run evals against live production data or dev datasets

<img width="1723" alt="develop-ui-requests-2" src="https://github.com/athina-ai/athina-evals/assets/7515552/f64a8701-a692-4616-817d-376207e8284b">

<br />

---

<br /><br />

### Need Production Monitoring and Evals? We've got you covered...

- Athina eval runs automatically write into Athina Dashboard, so you can view results and analytics in a beautiful UI.
- Athina track your experiments automatically, so you can view a historical record of previous eval runs.
- Athina calculates analytics segmented at every level possible, so you can view and compare your model performance at very granular levels.

![Athina Observe Platform](https://athina.ai/athina-dashboard-3.png)

## About Athina

Athina is building an end-to-end LLM monitoring and evaluation platform.

[Website](https://athina.ai) | [Demo Video](https://www.loom.com/share/d9ef2c62e91b46769a39c42bb6669834?sid=711df413-0adb-4267-9708-5f29cef929e3)

Contact us at [hello@athina.ai](mailto:hello@athina.ai) for any questions about the eval library.
