[Documentation](https://docs.athina.ai/evals) | [Watch Demo Video →](https://www.loom.com/share/10e37f1ba11242ac8c97902edd2fa61e)

## Overview

**Athina is an open-source library with plug-and-play preset evals designed to help engineers systematically improve their LLM reliability and performance through eval-driven-development.**

It's difficult to know if your LLM response is good or bad. Most developers start out by simply eyeballing the responses. This is fine when you're building a prototype and testing on 5-10 examples.

But once you optimize for reliability in production, this method breaks down.

Evals can help you:

- Detect regressions
- Measure performance of model (as defined by your goals)
- A/B test different models and prompts rapidly
- Monitor production data with confidence
- Run quantifiable experiments against ambiguous conversations

_Think of evals like unit tests for your LLM app._

<br />

## Quick Start

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

For more details, see this guide on [running evals](/evals/running_evals).

## <br />

<br />

### Why should I use Athina's Evals instead of writing my own?

You could build your own eval system from scratch, but here's why Athina might be better for you:

- Athina provides you with plug-and-play preset evals that have been well-tested
- Athina evals can run on both development and production, giving you consistent metrics for evaluating model performance and drift.
- Athina removes the need for your team to write boilerplate loaders, implement LLMs, normalize data formats, etc
- Athina offers a modular, extensible framework for writing and running evals
- Athina calculate analytics like pass rate and flakiness, and allows you to batch run evals against live production data or dev datasets

![Athina Evals Platform](https://docs.athina.ai/eval_results.png)

### Need Production Monitoring and Evals? We've got you covered...

- Athina eval runs automatically write into Athina Dashboard, so you can view results and analytics in a beautiful UI.
- Athina track your experiments automatically, so you can view a historical record of previous eval runs.
- Athina calculates analytics segmented at every level possible, so you can view and compare your model performance at very granular levels.

![Athina Observe Platform](https://athina.ai/athina-dashboard-3.png)

## About Athina

Athina is building an end-to-end LLM monitoring and evaluation platform.

[Website](https://athina.ai) | [Demo Video](https://www.loom.com/share/d9ef2c62e91b46769a39c42bb6669834?sid=711df413-0adb-4267-9708-5f29cef929e3)

Contact us at [hello@athina.ai](mailto:hello@athina.ai) for any questions about the eval library.
