#!/usr/bin/env python3
"""
Unit tests for the Hypermetric CLI tool
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import tempfile

# Import the module to test
import hypermetric


class TestHypermetric(unittest.TestCase):
    """Test cases for the Hypermetric CLI tool functions"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary custom prompts file
        self.temp_prompts = tempfile.NamedTemporaryFile(delete=False, mode='w')
        json.dump(["Test prompt 1", "Test prompt 2"], self.temp_prompts)
        self.temp_prompts.close()

        # Mock API key
        os.environ["HYPERBOLIC_API_KEY"] = "test_api_key"

    def tearDown(self):
        """Clean up after tests"""
        os.unlink(self.temp_prompts.name)
        if "HYPERBOLIC_API_KEY" in os.environ:
            del os.environ["HYPERBOLIC_API_KEY"]

    def test_parse_arguments(self):
        """Test argument parsing"""
        with patch('sys.argv', ['hypermetric.py', 'model1', 'model2']):
            args = hypermetric.parse_arguments()
            self.assertEqual(args.model1, 'model1')
            self.assertEqual(args.model2, 'model2')
            self.assertEqual(args.runs, 3)  # Default value
            self.assertEqual(args.prompt_set, 'custom')  # Default value

    def test_get_model_pricing(self):
        """Test model pricing lookup"""
        # Test known model
        pricing = hypermetric.get_model_pricing("deepseek-ai/DeepSeek-V3-0324")
        self.assertEqual(pricing["input"], 0.0028)
        self.assertEqual(pricing["output"], 0.0038)

        # Test unknown model (should return default values)
        pricing = hypermetric.get_model_pricing("unknown-model")
        self.assertEqual(pricing["input"], 0.0025)
        self.assertEqual(pricing["output"], 0.0035)

    def test_get_test_prompts(self):
        """Test prompt retrieval"""
        # Test default prompt set
        prompts = hypermetric.get_test_prompts("mmlu")
        self.assertEqual(len(prompts), 5)
        self.assertTrue(
            "Explain the difference between RAM and ROM in computing." in
            prompts)

        # Test custom prompts from file
        prompts = hypermetric.get_test_prompts("custom",
                                               self.temp_prompts.name)
        self.assertEqual(prompts, ["Test prompt 1", "Test prompt 2"])

    def test_calculate_consistency(self):
        """Test consistency calculation"""
        # Test identical responses
        results = {
            "responses": [["Hello world", "Hello world", "Hello world"]]
        }
        consistency = hypermetric.calculate_consistency(results)
        self.assertAlmostEqual(consistency, 100.0)

        # Test completely different responses
        results = {
            "responses": [["Hello world", "Goodbye moon", "Testing 123"]]
        }
        consistency = hypermetric.calculate_consistency(results)
        self.assertLess(consistency, 100.0)

    def test_calculate_cost(self):
        """Test cost calculation"""
        model_pricing = {"input": 0.0025, "output": 0.0035}
        tokens = {"input_tokens": 10, "output_tokens": 100}

        cost = hypermetric.calculate_cost(model_pricing, tokens)

        self.assertEqual(cost["cost_per_1k_input"], 0.0025)
        self.assertEqual(cost["cost_per_1k_output"], 0.0035)
        self.assertAlmostEqual(cost["input_cost"], 0.000025)
        self.assertAlmostEqual(cost["output_cost"], 0.00035)
        self.assertAlmostEqual(cost["total_cost"], 0.000375)

    @patch('requests.post')
    def test_benchmark_model(self, mock_post):
        """Test model benchmarking"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        mock_post.return_value = mock_response

        # Run benchmark with minimal test data
        results = hypermetric.benchmark_model("test-model", ["Test prompt"],
                                              runs=1)

        # Verify API was called
        mock_post.assert_called_once()

        # Check results structure
        self.assertTrue("time_to_first_token" in results)
        self.assertTrue("total_latency" in results)
        self.assertTrue("tokens_per_second" in results)
        self.assertTrue("input_tokens" in results)
        self.assertTrue("output_tokens" in results)
        self.assertTrue("total_tokens" in results)
        self.assertTrue("consistency" in results)

        # Verify token counts
        self.assertEqual(results["input_tokens"], 10)
        self.assertEqual(results["output_tokens"], 20)
        self.assertEqual(results["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
