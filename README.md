## Overview

**`athina-evals` is an framework to help you quickly set up evaluations and monitoring for your LLM-powered application**

It's difficult to know if your LLM response is good or bad. Most developers start out by simply eyeballing the responses. This is fine when you're building a prototype and testing on 5-10 examples.

But once you optimize for reliability in production, this method breaks down.

Evals can help you:

- Detect regressions
- Measure performance of model (as defined by your goals)
- A/B test different models and prompts rapidly
- Monitor production data with confidence
- Run quantifiable experiments against ambiguous conversations

_Think of evals like unit tests for your LLM app._

## Documentation

See [https://docs.athina.ai](https://docs.athina.ai) for the complete documentation.

## Quick Start

#### 1. Install the package

```
pip install athina-evals
```

#### 2. Get an Athina API key

Sign up at [athina.ai](https://athina.ai) to get an API key.

_(free, and only takes about 30 seconds)_

#### 3. Set API keys

```python
from athina.keys import AthinaApiKey, OpenAiApiKey

OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
```

#### 4. Run evals

```python
# Load the data from CSV, JSON, Athina or Dictionary
dataset = RagLoader().load_json(json_file)

# Run the DoesResponseAnswerQuery evaluator on the dataset
DoesResponseAnswerQuery().run_batch(data=dataset)
```

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
