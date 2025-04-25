# HyperMetric

A command-line tool for comparing LLM models on the Hyperbolic platform.

## About

HyperMetric is a powerful benchmarking tool that helps developers make informed decisions when selecting LLM models on the Hyperbolic platform. The tool compares models across three key dimensions:

- **Speed**: Time to first token, total latency, and tokens per second
- **Accuracy**: Response consistency and quality
- **Cost**: Token pricing and cost-performance ratio

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/hypermetric.git
cd hypermetric
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your Hyperbolic API key:
```
# Create a .env file with your API key
echo "HYPERBOLIC_API_KEY=your_key_here" > .env
```

## Usage

Basic comparison between two models:

```
python hypermetric.py meta-llama/Meta-Llama-3-70B-Instruct deepseek-ai/DeepSeek-V3-0324
```

Advanced options:

```
# Compare models with custom prompt set
python hypermetric.py model1 model2 --prompt-set humaneval

# Run more test iterations for better consistency measurement
python hypermetric.py model1 model2 --runs 5

# Use custom prompts from a JSON file
python hypermetric.py model1 model2 --custom-prompts my_prompts.json

# Output results in JSON format for further processing
python hypermetric.py model1 model2 --output json
```

### Example custom prompts file format (JSON):

```json
[
  "Write a function to validate email addresses in Python.",
  "Explain the concept of containerization to a junior developer.",
  "Compare and contrast microservices and monolithic architectures."
]
```

## Features

- **Comprehensive Metrics**: Measure performance across multiple dimensions
- **Customizable Tests**: Use built-in prompt sets or your own custom prompts
- **Beautiful Reports**: Rich console output with colored tables and formatting
- **JSON Output**: Export results for further analysis or integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
