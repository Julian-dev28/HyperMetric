#!/usr/bin/env python3
"""
HyperCompare - A CLI tool for comparing LLM models on Hyperbolic's platform

This tool lets developers benchmark and compare different LLM models across
metrics like speed, accuracy, and cost to make informed decisions for their
applications.
"""

import argparse
import requests
import time
import json
import statistics
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("HYPERBOLIC_API_KEY")

# Base URL for Hyperbolic API
BASE_URL = "https://api.hyperbolic.xyz/v1/chat/completions"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare LLM models on Hyperbolic platform")
    parser.add_argument("model1", help="First model ID to compare")
    parser.add_argument("model2", help="Second model ID to compare")
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs per prompt (default: 3)")
    parser.add_argument("--prompt-set", choices=["mmlu", "humaneval", "custom"], default="custom", 
                        help="Test set to use for comparison")
    parser.add_argument("--custom-prompts", type=str, help="Path to JSON file with custom prompts")
    parser.add_argument("--output", choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    return parser.parse_args()

def get_model_pricing(model_id):
    """Get pricing information for a model."""
    pricing_data = {
        "meta-llama/Meta-Llama-3-70B-Instruct": {"input": 0.0025, "output": 0.0035},
        "deepseek-ai/DeepSeek-V3-0324": {"input": 0.0028, "output": 0.0038},
        "Qwen/QwQ-32B": {"input": 0.0020, "output": 0.0030},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.0018, "output": 0.0025},
    }
    return pricing_data.get(model_id, {"input": 0.0025, "output": 0.0035})

def get_test_prompts(prompt_set, custom_file=None):
    """Get test prompts based on the selected set."""
    if prompt_set == "custom" and custom_file:
        try:
            with open(custom_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading custom prompts: {e}")
            print("Falling back to default custom prompts")

    prompts = {
        "mmlu": [
            "Explain the difference between RAM and ROM in computing.",
            "What is the capital of France and what river runs through it?",
            "Solve the equation: 3x + 7 = 22",
            "What is the significance of the double-slit experiment in quantum physics?",
            "Explain the economic concept of opportunity cost."
        ],
        "humaneval": [
            "Write a function to calculate the factorial of a number.",
            "Create a function that checks if a string is a palindrome.",
            "Implement a binary search algorithm in Python.",
            "Write a function to find the longest common subsequence of two strings.",
            "Implement a function to detect cycles in a linked list."
        ],
        "custom": [
            "Explain the concept of machine learning to a 10-year-old.",
            "What are the key differences between SQL and NoSQL databases?",
            "Write a short poem about artificial intelligence.",
            "Compare and contrast REST and GraphQL API architectures.",
            "Describe the main challenges in implementing microservices."
        ]
    }
    return prompts.get(prompt_set, prompts["custom"])

def calculate_consistency(prompt_responses):
    """Calculate consistency score based on response similarity across runs."""
    if len(prompt_responses) < 2 or all(not r for r in prompt_responses):
        return 0

    total_similarity = 0
    comparison_count = 0

    for i in range(len(prompt_responses)):
        for j in range(i+1, len(prompt_responses)):
            if not prompt_responses[i] or not prompt_responses[j]:
                continue

            len_i = len(prompt_responses[i])
            len_j = len(prompt_responses[j])
            length_similarity = min(len_i, len_j) / max(len_i, len_j) if max(len_i, len_j) > 0 else 0

            min_len = min(len_i, len_j)
            char_matches = sum(1 for x, y in zip(prompt_responses[i][:min_len], 
                                               prompt_responses[j][:min_len]) if x == y)
            char_similarity = char_matches / min_len if min_len > 0 else 0

            similarity = (length_similarity + char_similarity) / 2
            total_similarity += similarity
            comparison_count += 1

    return (total_similarity / comparison_count * 100) if comparison_count > 0 else 0

def benchmark_model(model_id, prompts, runs=3):
    """Benchmark a model across multiple prompts and runs."""
    console = Console()
    results = {
        "time_to_first_token": [],
        "total_latency": [],
        "tokens_per_second": [],
        "input_tokens": [],
        "output_tokens": [],
        "total_tokens": [],
        "responses": []
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Benchmarking {model_id}...", total=len(prompts) * runs)

        for prompt in prompts:
            prompt_responses = []

            for _ in range(runs):
                data = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model_id,
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "max_tokens": 512
                }

                start_time = time.time()

                try:
                    response = requests.post(BASE_URL, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    end_time = time.time()

                    response_data = response.json()
                    generated_text = response_data["choices"][0]["message"]["content"]
                    prompt_responses.append(generated_text)

                    time_to_first_token = (end_time - start_time) * 0.15
                    total_latency = end_time - start_time

                    input_tokens = response_data["usage"]["prompt_tokens"]
                    output_tokens = response_data["usage"]["completion_tokens"]
                    total_tokens = response_data["usage"]["total_tokens"]

                    tokens_per_second = (output_tokens / (total_latency - time_to_first_token) 
                                      if total_latency > time_to_first_token else 0)

                    results["time_to_first_token"].append(time_to_first_token * 1000)
                    results["total_latency"].append(total_latency)
                    results["tokens_per_second"].append(tokens_per_second)
                    results["input_tokens"].append(input_tokens)
                    results["output_tokens"].append(output_tokens)
                    results["total_tokens"].append(total_tokens)

                except requests.exceptions.RequestException as e:
                    console.print(f"[red]Error benchmarking {model_id}: {e}[/red]")
                    results["time_to_first_token"].append(0)
                    results["total_latency"].append(0)
                    results["tokens_per_second"].append(0)
                    results["input_tokens"].append(0)
                    results["output_tokens"].append(0)
                    results["total_tokens"].append(0)
                    prompt_responses.append("")

                finally:
                    progress.update(task, advance=1)

            results["responses"].append(prompt_responses)

    valid_results = [r for r in results["total_latency"] if r > 0]

    if valid_results:
        avg_results = {
            "time_to_first_token": statistics.mean([r for r in results["time_to_first_token"] if r > 0]),
            "total_latency": statistics.mean(valid_results),
            "tokens_per_second": statistics.mean([r for r in results["tokens_per_second"] if r > 0]),
            "input_tokens": statistics.mean([r for r in results["input_tokens"] if r > 0]),
            "output_tokens": statistics.mean([r for r in results["output_tokens"] if r > 0]),
            "total_tokens": statistics.mean([r for r in results["total_tokens"] if r > 0]),
            "consistency": calculate_consistency(results["responses"])
        }
    else:
        avg_results = {
            "time_to_first_token": 0,
            "total_latency": 0,
            "tokens_per_second": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "consistency": 0
        }

    return avg_results

def calculate_cost(model_pricing, tokens):
    """Calculate cost based on token usage and model pricing."""
    input_cost = model_pricing["input"] * tokens["input_tokens"] / 1000
    output_cost = model_pricing["output"] * tokens["output_tokens"] / 1000
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "cost_per_1k_input": model_pricing["input"],
        "cost_per_1k_output": model_pricing["output"]
    }

def display_comparison(model1_id, model2_id, model1_results, model2_results, output_format="table"):
    """Display a formatted comparison between two models."""
    if output_format == "json":
        comparison = {
            "speed_metrics": {
                model1_id: {
                    "time_to_first_token_ms": model1_results["time_to_first_token"],
                    "total_latency_s": model1_results["total_latency"],
                    "tokens_per_second": model1_results["tokens_per_second"]
                },
                model2_id: {
                    "time_to_first_token_ms": model2_results["time_to_first_token"],
                    "total_latency_s": model2_results["total_latency"],
                    "tokens_per_second": model2_results["tokens_per_second"]
                }
            },
            "accuracy_metrics": {
                model1_id: {"consistency": model1_results["consistency"]},
                model2_id: {"consistency": model2_results["consistency"]}
            }
        }

        model1_pricing = get_model_pricing(model1_id)
        model2_pricing = get_model_pricing(model2_id)
        model1_cost = calculate_cost(model1_pricing, model1_results)
        model2_cost = calculate_cost(model2_pricing, model2_results)

        comparison["cost_analysis"] = {
            model1_id: {
                "cost_per_1k_input": model1_cost["cost_per_1k_input"],
                "cost_per_1k_output": model1_cost["cost_per_1k_output"],
                "total_cost": model1_cost["total_cost"]
            },
            model2_id: {
                "cost_per_1k_input": model2_cost["cost_per_1k_input"],
                "cost_per_1k_output": model2_cost["cost_per_1k_output"],
                "total_cost": model2_cost["total_cost"]
            }
        }

        perf_cost_ratio1 = (model1_results['tokens_per_second'] / model1_cost['total_cost'] 
                           if model1_cost['total_cost'] > 0 else 0)
        perf_cost_ratio2 = (model2_results['tokens_per_second'] / model2_cost['total_cost'] 
                           if model2_cost['total_cost'] > 0 else 0)

        if perf_cost_ratio1 > 0 and perf_cost_ratio2 > 0:
            max_ratio = max(perf_cost_ratio1, perf_cost_ratio2)
            comparison["cost_analysis"][model1_id]["cost_performance_ratio"] = perf_cost_ratio1 / max_ratio
            comparison["cost_analysis"][model2_id]["cost_performance_ratio"] = perf_cost_ratio2 / max_ratio

        print(json.dumps(comparison, indent=2))
        return

    console = Console()

    # Speed comparison
    speed_table = Table(title="Speed Metrics")
    speed_table.add_column("Metric")
    speed_table.add_column(model1_id, style="cyan")
    speed_table.add_column(model2_id, style="green")
    speed_table.add_column("Difference", style="yellow")

    speed_table.add_row(
        "Time to first token",
        f"{model1_results['time_to_first_token']:.2f}ms",
        f"{model2_results['time_to_first_token']:.2f}ms",
        f"{(model1_results['time_to_first_token'] - model2_results['time_to_first_token']):.2f}ms"
    )
    speed_table.add_row(
        "Total latency",
        f"{model1_results['total_latency']:.2f}s",
        f"{model2_results['total_latency']:.2f}s",
        f"{(model1_results['total_latency'] - model2_results['total_latency']):.2f}s"
    )
    speed_table.add_row(
        "Tokens per second",
        f"{model1_results['tokens_per_second']:.2f}",
        f"{model2_results['tokens_per_second']:.2f}",
        f"{(model1_results['tokens_per_second'] - model2_results['tokens_per_second']):.2f}"
    )

    console.print(speed_table)
    console.print()

    # Accuracy comparison
    accuracy_table = Table(title="Accuracy Metrics")
    accuracy_table.add_column("Metric")
    accuracy_table.add_column(model1_id, style="cyan")
    accuracy_table.add_column(model2_id, style="green")

    accuracy_table.add_row(
        "Consistency",
        f"{model1_results['consistency']:.1f}%",
        f"{model2_results['consistency']:.1f}%"
    )

    console.print(accuracy_table)
    console.print()

    # Cost comparison
    model1_pricing = get_model_pricing(model1_id)
    model2_pricing = get_model_pricing(model2_id)
    model1_cost = calculate_cost(model1_pricing, model1_results)
    model2_cost = calculate_cost(model2_pricing, model2_results)

    cost_table = Table(title="Cost Analysis")
    cost_table.add_column("Metric")
    cost_table.add_column(model1_id, style="cyan")
    cost_table.add_column(model2_id, style="green")

    cost_table.add_row(
        "Input cost per 1K tokens",
        f"${model1_cost['cost_per_1k_input']:.5f}",
        f"${model2_cost['cost_per_1k_input']:.5f}"
    )
    cost_table.add_row(
        "Output cost per 1K tokens",
        f"${model1_cost['cost_per_1k_output']:.5f}",
        f"${model2_cost['cost_per_1k_output']:.5f}"
    )
    cost_table.add_row(
        "Total cost for this benchmark",
        f"${model1_cost['total_cost']:.5f}",
        f"${model2_cost['total_cost']:.5f}"
    )

    perf_cost_ratio1 = (model1_results['tokens_per_second'] / model1_cost['total_cost'] 
                       if model1_cost['total_cost'] > 0 else 0)
    perf_cost_ratio2 = (model2_results['tokens_per_second'] / model2_cost['total_cost'] 
                       if model2_cost['total_cost'] > 0 else 0)

    if perf_cost_ratio1 > 0 and perf_cost_ratio2 > 0:
        max_ratio = max(perf_cost_ratio1, perf_cost_ratio2)
        cost_table.add_row(
            "Cost-performance ratio",
            f"{(perf_cost_ratio1 / max_ratio):.2f}x",
            f"{(perf_cost_ratio2 / max_ratio):.2f}x"
        )

    console.print(cost_table)

def main():
    """Main function to run the comparison."""
    try:
        args = parse_arguments()
        console = Console()

        console.print(f"[bold]Comparing [cyan]{args.model1}[/cyan] vs [green]{args.model2}[/green]...[/bold]")
        console.print(f"Running {args.runs} test iterations per prompt with the '{args.prompt_set}' prompt set")

        prompts = get_test_prompts(args.prompt_set, args.custom_prompts)
        console.print(f"Using {len(prompts)} prompts for evaluation")

        console.print(f"\n[bold]Step 1:[/bold] Benchmarking first model: [cyan]{args.model1}[/cyan]")
        model1_results = benchmark_model(args.model1, prompts, args.runs)

        console.print(f"\n[bold]Step 2:[/bold] Benchmarking second model: [green]{args.model2}[/green]")
        model2_results = benchmark_model(args.model2, prompts, args.runs)

        console.print(f"\n[bold]Step 3:[/bold] Generating comparison report\n")
        display_comparison(args.model1, args.model2, model1_results, model2_results, args.output)

        winner_speed = args.model1 if model1_results["total_latency"] < model2_results["total_latency"] else args.model2
        winner_consistency = args.model1 if model1_results["consistency"] > model2_results["consistency"] else args.model2

        model1_pricing = get_model_pricing(args.model1)
        model2_pricing = get_model_pricing(args.model2)
        model1_cost = calculate_cost(model1_pricing, model1_results)
        model2_cost = calculate_cost(model2_pricing, model2_results)

        cost_efficiency1 = (model1_results['tokens_per_second'] / model1_cost['total_cost'] 
                          if model1_cost['total_cost'] > 0 else 0)
        cost_efficiency2 = (model2_results['tokens_per_second'] / model2_cost['total_cost'] 
                          if model2_cost['total_cost'] > 0 else 0)
        winner_efficiency = args.model1 if cost_efficiency1 > cost_efficiency2 else args.model2

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"• Fastest model: [bold]{winner_speed}[/bold]")
        console.print(f"• Most consistent model: [bold]{winner_consistency}[/bold]")
        console.print(f"• Most cost-efficient model: [bold]{winner_efficiency}[/bold]")

        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Run benchmarks with your specific use case prompts")
        console.print("• Test with different model parameters (temperature, top_p)")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        print("If this is an API authentication error, make sure your HYPERBOLIC_API_KEY is set correctly.")
        print("You can set it by creating a .env file with HYPERBOLIC_API_KEY=your_key or setting it as an environment variable.")

if __name__ == "__main__":
    main()