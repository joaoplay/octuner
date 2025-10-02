"""
This file demonstrates a multistep sentiment analysis pipeline using MultiProviderTunableLLM for provider optimization.

It showcases how the optimizer can automatically discover the best combination of providers, models, and parameters for
each step in the pipeline. This is meant to be a realistic example, which requires API keys whenever you are including
providers like OpenAI, Gemini, etc.

Multiple logs are included to show the optimization process, including the search space and best parameters found.

Remember that you are free to set your own configuration file, and use your own model-chain design.
The TunableSentimentAnalyzer class is just an example of how to structure a multistep pipeline with tunable LLMs.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from octuner import MultiProviderTunableLLM, AutoTuner, apply_best


@dataclass
class SentimentResult:
    """Result from sentiment analysis pipeline."""
    initial_sentiment: str
    confidence_score: float
    reasoning: str
    final_sentiment: str
    final_confidence: float


class TunableSentimentAnalyzer:
    """
    Multistep sentiment analysis using MultiProviderTunableLLM for provider optimization.
    
    This component demonstrates how MultiProviderTunableLLM can automatically discover
    the best combination of providers, models, and parameters for each step
    in a multi-call LLM pipeline.
    
    Key Features:
    - Multi-provider optimization (OpenAI, Gemini)
    - Tunable websearch: The optimizer decides when websearch improves performance
    - Context-aware prompts: Work effectively both with and without websearch
    - analyze_sentiment(): Unified method that leverages websearch when beneficial
    
    Websearch Use Cases:
    - Information gathering: Enable websearch for context about products, companies, events
    - Pure analysis: Disable websearch when external context doesn't help
    - Optimizer control: Let the optimizer automatically decide based on performance
    """

    def __init__(self, llm_config_file: str, provider_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                 user_location: Optional[Dict[str, str]] = None):
        """
        Initialize the tunable sentiment analyzer. This is an example of a multi-model chain:
            1. Classifier LLM: Classifies sentiment as positive, negative, or neutral
            2. Confidence LLM: Rates confidence in the classification
            3. Reasoning LLM: Provides detailed reasoning and final sentiment decision
        
        Args:
            llm_config_file: Path to YAML configuration file defining LLM capabilities
            provider_configs: Configuration for each provider (API keys, etc.)
                            If None, will auto-detect from environment variables
            user_location: Optional user location for websearch. Can be:
                          - Dict: {"city": "Coimbra", "country": "PT"} (will be wrapped in proper format)
                          - Full format: {"approximate": {"type": "approximate", "city": "Coimbra", "country": "PT"}}
                          This is passed to OpenAI's websearch tool for localized results
        """
        self.llm_config_file = llm_config_file

        # Auto-detect provider configurations from environment
        if provider_configs is None:
            provider_configs = self._auto_detect_provider_configs()

        # Add user_location to OpenAI provider config if provided
        if user_location and "openai" in provider_configs:
            provider_configs["openai"]["user_location"] = user_location

        self.provider_configs = provider_configs

        # Classifies sentiment
        self.classifier_llm = MultiProviderTunableLLM(
            config_file=llm_config_file,
            default_provider="gemini",
            provider_configs=provider_configs
        )

        # Rates confidence in classification
        self.confidence_llm = MultiProviderTunableLLM(
            config_file=llm_config_file,
            default_provider="gemini",
            provider_configs=provider_configs
        )

        # Provides detailed reasoning and final decision
        self.reasoning_llm = MultiProviderTunableLLM(
            config_file=llm_config_file,
            default_provider="gemini",
            provider_configs=provider_configs
        )

    @staticmethod
    def _auto_detect_provider_configs() -> Dict[str, Dict[str, Any]]:
        """
        Auto-detect provider configurations from environment variables.
        """
        configs = {}

        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            configs['openai'] = {'api_key': os.getenv('OPENAI_API_KEY')}

        # Gemini/Google
        if os.getenv('GOOGLE_API_KEY'):
            configs['gemini'] = {'api_key': os.getenv('GOOGLE_API_KEY')}

        return configs

    def force_websearch(self):
        """
        Force enable or disable websearch for all steps (for testing).

        This method is useful for demonstration purposes to see the impact of websearch.
        """
        self.classifier_llm.use_websearch = True
        self.confidence_llm.use_websearch = False
        self.reasoning_llm.use_websearch = False

    def reset_websearch(self):
        """
        Reset websearch settings to default (optimizer-controlled).

        This method disables websearch for all steps, allowing the optimizer to decide.
        """
        self.classifier_llm.use_websearch = False
        self.confidence_llm.use_websearch = False
        self.reasoning_llm.use_websearch = False

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using a multistep approach with optimizable providers.
        
        This method is called at each step of the optimization process. Note that it is used in the entrypoint
        parameter of AutoTuner.from_component().

        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with detailed analysis
        """
        # Step 1: Context-aware sentiment classification 
        # Prompt designed to work well both with and without websearch
        classifier_prompt = f"""
        Analyze the sentiment of the following text and classify it as "positive", "negative", or "neutral".
        
        If this text mentions specific products, companies, events, or entities, consider any relevant 
        background information that might affect sentiment interpretation.
        
        Text: {text}
        
        Respond with only the sentiment label: positive, negative, or neutral.
        """

        try:
            classifier_response = self.classifier_llm.call(
                classifier_prompt,
                "You are a sentiment classification expert. Use available context when helpful, but focus on the core sentiment. Respond only with the sentiment label."
                # Note: use_websearch now controlled by tuning system
            )
            initial_sentiment = classifier_response.text.strip().lower()

            # Ensure valid sentiment
            if initial_sentiment not in ["positive", "negative", "neutral"]:
                # Extract sentiment from response if it contains extra text
                for sentiment in ["positive", "negative", "neutral"]:
                    if sentiment in initial_sentiment:
                        initial_sentiment = sentiment
                        break
                else:
                    initial_sentiment = "neutral"  # Default fallback

        except Exception as e:
            print(f"Classification step failed: {e}")
            initial_sentiment = "neutral"

        # Step 2: Context-aware confidence assessment
        confidence_prompt = f"""
        Given this text and initial sentiment classification, rate the confidence from 0.0 to 1.0.
        Consider factors like clarity, emotional intensity, ambiguity, and any relevant context 
        about mentioned entities.
        
        Text: {text}
        Initial sentiment: {initial_sentiment}
        
        Respond with only a number between 0.0 and 1.0.
        """

        try:
            confidence_response = self.confidence_llm.call(
                confidence_prompt,
                "You assess confidence in sentiment classifications. Use available context to inform your confidence rating. Respond only with a decimal number."
                # Note: use_websearch now controlled by tuning system
            )

            # Extract confidence score from response
            confidence_text = confidence_response.text.strip()
            # Extract numeric value from response (handle various formats)
            import re
            confidence_match = re.search(r'(\d*\.?\d+)', confidence_text)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))
                confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0,1]
            else:
                confidence_score = 0.5  # Default if no number found

        except Exception as e:
            print(f"Confidence assessment step failed: {e}")
            confidence_score = 0.5

        # Step 3: Context-aware reasoning and final decision
        reasoning_prompt = f"""
        Provide detailed sentiment analysis with reasoning.
        
        Text: {text}
        Initial assessment: {initial_sentiment} (confidence: {confidence_score})
        
        Consider any relevant context about mentioned products, companies, or events in your analysis.
        
        Format your response as:
        Reasoning: [detailed reasoning incorporating any available context]
        Final sentiment: [positive/negative/neutral]
        Final confidence: [0.0-1.0]
        """

        try:
            reasoning_response = self.reasoning_llm.call(
                reasoning_prompt,
                "You provide detailed sentiment analysis with reasoning. Incorporate available context when relevant. Follow the exact format requested."
                # Note: use_websearch now controlled by tuning system
            )

            # Parse the reasoning response
            reasoning_text = reasoning_response.text
            reasoning = ""
            final_sentiment = initial_sentiment
            final_confidence = confidence_score

            lines = reasoning_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Reasoning:"):
                    reasoning = line[10:].strip()
                elif line.startswith("Final sentiment:"):
                    final_sentiment_raw = line[16:].strip().lower()
                    # Ensure valid sentiment
                    if final_sentiment_raw in ["positive", "negative", "neutral"]:
                        final_sentiment = final_sentiment_raw
                elif line.startswith("Final confidence:"):
                    try:
                        import re
                        conf_text = line[17:].strip()
                        conf_match = re.search(r'(\d*\.?\d+)', conf_text)
                        if conf_match:
                            final_confidence = float(conf_match.group(1))
                            final_confidence = max(0.0, min(1.0, final_confidence))
                    except ValueError:
                        pass

            # If reasoning wasn't extracted, use the full response
            if not reasoning:
                reasoning = reasoning_text

        except Exception as e:
            print(f"Reasoning step failed: {e}")
            reasoning = f"Error in reasoning step: {e}"
            final_sentiment = initial_sentiment
            final_confidence = confidence_score

        return SentimentResult(
            initial_sentiment=initial_sentiment,
            confidence_score=confidence_score,
            reasoning=reasoning,
            final_sentiment=final_sentiment,
            final_confidence=final_confidence
        )

    def get_provider_summary(self) -> Dict[str, str]:
        """Get current provider configuration for each step."""
        summary = {}
        for step_name, llm in [
            ("classifier", self.classifier_llm),
            ("confidence", self.confidence_llm),
            ("reasoning", self.reasoning_llm)
        ]:
            info = llm.get_current_provider_info()
            provider = info.get('provider', 'unknown')
            model = info.get('model', 'unknown')
            summary[step_name] = f"{provider}:{model}"
        return summary


# Golden dataset for sentiment analysis - expanded version
def create_sentiment_dataset() -> List[Dict[str, Any]]:
    """
    Create a comprehensive golden dataset for sentiment analysis.
    
    This dataset includes various types of text with known sentiment labels,
    representing common real-world scenarios and edge cases.
    """
    return [
        # Clear positive examples
        {
            "input": "I absolutely love this product! It exceeded all my expectations and works perfectly.",
            "target": SentimentResult(
                initial_sentiment="positive",
                confidence_score=0.9,
                reasoning="Strong positive language with words like 'absolutely love' and 'exceeded expectations'",
                final_sentiment="positive",
                final_confidence=0.95
            )
        },
        {
            "input": "Amazing! This changed my life in the best way possible. Highly recommend to everyone!",
            "target": SentimentResult(
                initial_sentiment="positive",
                confidence_score=0.95,
                reasoning="Extremely positive language with life-changing impact and recommendation",
                final_sentiment="positive",
                final_confidence=0.98
            )
        },
        {
            "input": "Fantastic quality and great value for money. Very satisfied with my purchase!",
            "target": SentimentResult(
                initial_sentiment="positive",
                confidence_score=0.9,
                reasoning="Multiple positive indicators: fantastic, great, very satisfied",
                final_sentiment="positive",
                final_confidence=0.92
            )
        },

        # Clear negative examples
        {
            "input": "This is the worst experience I've ever had. Completely disappointed and frustrated.",
            "target": SentimentResult(
                initial_sentiment="negative",
                confidence_score=0.9,
                reasoning="Clear negative sentiment with superlatives like 'worst' and emotional words",
                final_sentiment="negative",
                final_confidence=0.95
            )
        },
        {
            "input": "Terrible quality, broke after one day. Waste of money and time.",
            "target": SentimentResult(
                initial_sentiment="negative",
                confidence_score=0.9,
                reasoning="Multiple negative indicators: terrible, broke, waste",
                final_sentiment="negative",
                final_confidence=0.92
            )
        },
        {
            "input": "Poor customer service and the product doesn't work as advertised. Very disappointed.",
            "target": SentimentResult(
                initial_sentiment="negative",
                confidence_score=0.85,
                reasoning="Specific complaints about service and functionality with clear disappointment",
                final_sentiment="negative",
                final_confidence=0.88
            )
        },

        # Neutral examples
        {
            "input": "The weather today is okay, not particularly good or bad.",
            "target": SentimentResult(
                initial_sentiment="neutral",
                confidence_score=0.8,
                reasoning="Explicitly neutral language with balanced assessment",
                final_sentiment="neutral",
                final_confidence=0.85
            )
        },
        {
            "input": "The service was acceptable. Nothing special but it got the job done.",
            "target": SentimentResult(
                initial_sentiment="neutral",
                confidence_score=0.7,
                reasoning="Neutral to slightly positive but lukewarm assessment",
                final_sentiment="neutral",
                final_confidence=0.75
            )
        },
        {
            "input": "Standard delivery time and the product matches the description exactly.",
            "target": SentimentResult(
                initial_sentiment="neutral",
                confidence_score=0.8,
                reasoning="Factual description without emotional language",
                final_sentiment="neutral",
                final_confidence=0.8
            )
        },

        # Ambiguous/mixed examples (challenging cases)
        {
            "input": "I'm not sure how I feel about this. It has some good points but also some issues.",
            "target": SentimentResult(
                initial_sentiment="neutral",
                confidence_score=0.6,
                reasoning="Ambiguous sentiment with both positive and negative aspects mentioned",
                final_sentiment="neutral",
                final_confidence=0.65
            )
        },
        {
            "input": "Great design but poor build quality. Looks amazing but feels cheap.",
            "target": SentimentResult(
                initial_sentiment="negative",
                confidence_score=0.6,
                reasoning="Mixed review with positive design but negative quality, overall negative",
                final_sentiment="negative",
                final_confidence=0.6
            )
        },
        {
            "input": "It's fine I guess. Could be better but could also be worse. Nothing to complain about.",
            "target": SentimentResult(
                initial_sentiment="neutral",
                confidence_score=0.8,
                reasoning="Indifferent tone with balanced perspective and no strong feelings",
                final_sentiment="neutral",
                final_confidence=0.8
            )
        }
    ]


def sentiment_metric(output: SentimentResult, target: SentimentResult) -> float:
    """
    Enhanced evaluation metric for sentiment analysis results.
    
    This metric considers multiple aspects:
    - Correct final sentiment classification (primary importance)
    - Accuracy of confidence scores (secondary importance)
    - Quality of reasoning (tertiary importance)
    
    Args:
        output: Predicted sentiment result
        target: Expected sentiment result
        
    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    # Primary metric: correct final sentiment (75% weight)
    if output.final_sentiment == target.final_sentiment:
        score += 0.75

    # Confidence accuracy (20% weight)
    confidence_diff = abs(output.final_confidence - target.final_confidence)
    confidence_score = max(0.0, 1.0 - confidence_diff)
    score += 0.2 * confidence_score

    # Reasoning quality (5% weight) - configurable method
    if target.reasoning and output.reasoning:
        # Simple word overlap (fast)
        target_keywords = set(target.reasoning.lower().split())
        output_keywords = set(output.reasoning.lower().split())

        if target_keywords and output_keywords:
            # Check for overlapping concepts
            overlap = len(target_keywords & output_keywords)
            reasoning_score = min(1.0, overlap / max(len(target_keywords), 1))
            score += 0.05 * reasoning_score

    return min(1.0, score)


def main(config_file: str):
    """
    Run the MultiProviderTunableLLM sentiment analysis optimization example.
    
    Args:
        config_file: Path to YAML configuration file
    """
    print("Octuner - MultiProviderTunableLLM Sentiment Analysis Example")
    print("=" * 60)
    print(f"Configuration: {config_file}")
    print()

    # Check for available API keys
    api_keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Gemini': os.getenv('GOOGLE_API_KEY')
    }

    available_providers = []
    for provider, key in api_keys.items():
        if key:
            available_providers.append(provider)
            print(f"✓ {provider} API key found")
        else:
            print(f"✗ {provider} API key not found")

    if not available_providers:
        print("\nWarning: No API keys found. This example requires at least one provider.")
        print("Set API keys with:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export GOOGLE_API_KEY='your-gemini-key'")
        print("\nProceeding with mock configuration for demonstration...")
    else:
        print(f"\nFound {len(available_providers)} provider(s): {', '.join(available_providers)}")

    # Create component and dataset
    # Example: Pass user location for localized websearch results
    # You can pass either a simple dict or the full format:
    user_location = {"type": "approximate", "city": "Manchester", "country": "UK"}
    analyzer = TunableSentimentAnalyzer(
        llm_config_file=config_file,
        user_location=user_location,
    )
    dataset = create_sentiment_dataset()

    print(f"\nDataset size: {len(dataset)} examples")
    print(f"User location for websearch: {user_location}")

    # Show initial provider configuration
    print("\nInitial provider configuration:")
    provider_summary = analyzer.get_provider_summary()
    for step, provider_model in provider_summary.items():
        print(f"  {step}: {provider_model}")

    # Create tuner with metric function that uses the selected reasoning metric
    def metric_with_reasoning(output, target):
        return sentiment_metric(output, target)

    tuner = AutoTuner.from_component(
        component=analyzer,
        entrypoint=lambda a, text: a.analyze_sentiment(text),
        dataset=dataset,
        metric=metric_with_reasoning,
    )

    # Build search space and show summary
    tuner.build_search_space()
    # Get a summary of the search space
    summary = tuner.get_search_space_summary()
    print(f"\nSearch space: {summary['total_parameters']} parameters")
    print("Parameter types:", summary['parameter_types'])
    print("Components:", list(summary['components'].keys()))

    # Print search space details, with the names of parameters
    print("\nSearch space details:")
    for param_path, param in tuner.search_space.items():
        print(f"  {param_path}: {param}")

    # Optional: focus on most impactful parameters for sentiment analysis
    tuner.include([
        "*.provider_model",  # Provider and model selection (most important)
        "*.temperature",  # Controls randomness
        "*.max_tokens",  # Response length
        "*.top_p",  # Nucleus sampling
        "*.use_websearch",  # Web search capability - important for context-aware analysis
        "*.search_context_size"  # Search context size - controls how much web context to include
    ]).exclude([
        "*.frequency_penalty",  # Less important for sentiment
        "*.presence_penalty"  # Less important for sentiment
    ])

    print(f"\nFocused search space: {len(tuner.search_space)} parameters")

    # Run tests before optimization
    print("\nTesting default configuration...")
    test_text = "This product is amazing and I love using it every day!"
    test_context_text = "The new Tesla Model S update has been released and I'm really impressed with the performance improvements!"

    # Test 1: Basic sentiment analysis (no websearch)
    print("\n1. Basic sentiment analysis (websearch explicitly disabled):")
    test_result = analyzer.analyze_sentiment(test_text)
    print(f"Input: '{test_text}'")
    print(f"Result: {test_result.final_sentiment} (confidence: {test_result.final_confidence:.2f})")
    print(f"Reasoning: {test_result.reasoning}")

    # Test 2: Sentiment analysis with websearch forced (for demonstration)
    print("\n2. Sentiment analysis with websearch enabled (demonstration):")
    # Temporarily force websearch for demonstration
    analyzer.force_websearch()
    test_result_context = analyzer.analyze_sentiment(test_context_text)
    # Reset websearch to default/optimized values
    analyzer.reset_websearch()
    print(f"Input: '{test_context_text}'")
    print(f"Result: {test_result_context.final_sentiment} (confidence: {test_result_context.final_confidence:.2f})")
    print(f"Reasoning: {test_result_context.reasoning}")

    print("\nStarting optimization...")
    print("This will test different provider/model combinations and may take several minutes...")

    result = tuner.search(
        max_trials=5,
        mode="pareto",  # Balance quality and cost
        replicates=1,
        timeout=300,  # 5 minutes timeout for optimization
        seed=42
    )

    print(f"\nOptimization completed!")
    print(f"Best quality: {result.best_trial.metrics.quality:.3f}")
    cost_str = f"${result.best_trial.metrics.cost:.6f}" if result.best_trial.metrics.cost else "N/A"
    latency_str = f"{result.best_trial.metrics.latency_ms:.1f}ms" if result.best_trial.metrics.latency_ms else "N/A"
    print(f"Best cost: {cost_str}")
    print(f"Best latency: {latency_str}")

    # Save best parameters
    result.save_best("best_sentiment_tunable_llm.yaml")
    print("\nBest parameters saved to 'best_sentiment_tunable_llm.yaml'")

    # Show best parameters
    print("\nBest parameters found:")
    for param_path, value in result.best_parameters.items():
        print(f"  {param_path}: {value}")

    # Demonstrate applying best parameters
    print("\nTesting optimized sentiment analyzer...")
    new_analyzer = TunableSentimentAnalyzer(config_file)
    apply_best(new_analyzer, "best_sentiment_tunable_llm.yaml")

    # Show optimized provider configuration
    print("\nOptimized provider configuration:")
    optimized_summary = new_analyzer.get_provider_summary()
    for step, provider_model in optimized_summary.items():
        print(f"  {step}: {provider_model}")

    # Test with the optimized configuration
    test_result_optimized = new_analyzer.analyze_sentiment(test_text)

    print(f"\nOptimized analysis for: '{test_text}'")
    print(f"Final sentiment: {test_result_optimized.final_sentiment}")
    print(f"Confidence: {test_result_optimized.final_confidence:.2f}")
    print(f"Reasoning: {test_result_optimized.reasoning}")

    # Show improvement comparison if we have both results
    try:
        original_score = sentiment_metric(test_result, test_result)  # Score against itself
        optimized_score = sentiment_metric(test_result_optimized, test_result)
        print(f"\nPerformance comparison on test case:")
        print(f"Original configuration score: {original_score:.3f}")
        print(f"Optimized configuration score: {optimized_score:.3f}")
        if optimized_score > original_score:
            print("✓ Optimization improved performance!")
        elif optimized_score == original_score:
            print("= Optimization maintained performance (cost/latency may be better)")
        else:
            print("⚠ Optimization focused on different test cases")
    except:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Octuner - MultiProviderTunableLLM Sentiment Analysis Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentiment_analysis_tunable_llm.py config_templates/task_specific.yaml
  python sentiment_analysis_tunable_llm.py config_templates/task_specific.yaml --reasoning-metric bleu
  python sentiment_analysis_tunable_llm.py config_templates/task_specific.yaml --reasoning-metric word_overlap

Available config templates:
  - config_templates/openai_basic.yaml (OpenAI only)
  - config_templates/gemini_basic.yaml (Gemini only with Google grounding tool)
  - config_templates/multi_provider.yaml (OpenAI + Gemini with web search)
  - config_templates/task_specific.yaml
        """
    )

    parser.add_argument("config_file", help="Path to YAML configuration file")

    args = parser.parse_args()

    main(args.config_file)
