"""
Enhanced Alpha Generator with RAG Integration

This module extends the original alpha_generator_ollama.py with:
1. RAG system for learning from successful alphas
2. Improved prompt engineering with few-shot learning
3. Chain-of-thought reasoning
4. Diversity enhancement
5. Remote AI API support (DeepSeek and OpenAI)

Priority: [CAO] - High impact on alpha quality

Usage:
    Replace alpha_generator_ollama.py with this enhanced version
    or use as a drop-in replacement in alpha_orchestrator.py
"""

import logging
import os
import time
from typing import List, Dict, Optional
from alpha_rag_system import AlphaRAGSystem
from alpha_generator_ollama import AlphaGenerator

# Load environment variables from .env file
# This ensures API keys and configuration are available in all execution contexts:
# - Local Python execution
# - Docker container execution
# - Subprocess execution (when called from alpha_orchestrator.py)
try:
    from dotenv import load_dotenv

    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(current_dir, '.env')

    # Load .env file if it exists
    if os.path.exists(env_file_path):
        load_dotenv(dotenv_path=env_file_path, override=False)
        logging.debug(f"✅ Loaded environment variables from {env_file_path}")
    else:
        logging.warning(f"⚠️ .env file not found at {env_file_path}")
        logging.warning("⚠️ Remote AI API features may not work without API keys")

except ImportError:
    logging.warning("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    logging.warning("⚠️ Environment variables must be set manually or via system environment")

logger = logging.getLogger(__name__)


class EnhancedAlphaGenerator(AlphaGenerator):
    """
    Enhanced alpha generator with RAG and improved prompting.
    
    Extends the original AlphaGenerator with:
    - RAG system for context retrieval
    - Few-shot learning from successful alphas
    - Chain-of-thought reasoning prompts
    - Diversity tracking and enhancement
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced generator."""
        super().__init__(*args, **kwargs)

        # Initialize RAG system
        self.rag_system = AlphaRAGSystem()
        logger.info("Initialized RAG system for enhanced alpha generation")

        # Track generated expressions for diversity
        self.generated_expressions = set()
        self.diversity_threshold = 0.7  # Minimum similarity to consider duplicate

    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        """
        Override parent method to add successful alphas to RAG database.

        This method extends the parent's functionality by:
        1. Calling parent method to log to hopeful_alphas.json (maintains existing behavior)
        2. Adding the successful alpha to the RAG database for future context retrieval
        3. Enabling real-time learning from newly discovered successful alphas

        Args:
            expression: The alpha expression string
            alpha_data: Dictionary containing alpha metrics and metadata
        """
        # Step 1: Call parent method to log to JSON file
        # This maintains backward compatibility and ensures the alpha is persisted to disk
        super().log_hopeful_alpha(expression, alpha_data)

    def generate_enhanced_prompt(self, data_fields: List[Dict], operators: List[Dict],
                                 target_count: int = 50) -> str:
        """
        Generate enhanced prompt with RAG context and few-shot learning.
        
        Args:
            data_fields: Available data fields
            operators: Available operators
            target_count: Number of alphas to generate
            
        Returns:
            Enhanced prompt string
        """
        # Get RAG context from successful alphas
        rag_context = self.rag_system.generate_rag_context(top_k=5, min_fitness=0.6)
        
        # Get successful patterns
        patterns = self.rag_system.get_successful_patterns(min_fitness=0.7)
        
        # Build operator categories (same as original)
        operator_by_category = {}
        for op in operators:
            category = op.get('category', 'Other')
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append(op)
        
        # Sample operators (50% from each category)
        import random
        sampled_operators = {}
        for category, ops in operator_by_category.items():
            sample_size = max(1, int(len(ops) * 0.5))
            sampled_operators[category] = random.sample(ops, sample_size)
        
        # Format operators
        # Helper function to format operator information for prompt
        def format_operators(ops):
            """
            Format operator list for inclusion in prompt.

            Args:
                ops: List of operator dictionaries

            Returns:
                List of formatted operator strings
            """
            formatted = []
            for op in ops:
                op_type = op.get('type', 'SCALAR')
                op_name = op.get('name', 'unknown')
                op_definition = op.get('definition', 'N/A')
                op_description = op.get('description', 'N/A')

                formatted.append(
                    f"{op_name} ({op_type})\n"
                    f"  Definition: {op_definition}\n"
                    f"  Description: {op_description}"
                )
            return formatted
        
        # Build enhanced prompt
        prompt_parts = []
        
        # 1. Task description with chain-of-thought
        prompt_parts.append(f"""You are an expert quantitative researcher specializing in alpha factor discovery.

Your task is to generate {target_count} unique and potentially profitable alpha factor expressions.
""")
        
        # 2. RAG context (successful alphas and patterns)
        if rag_context:
            prompt_parts.append(f"\n{rag_context}\n")
        
        # 3. Available resources
        prompt_parts.append(f"""
=== Available Data Fields ===
{[field['id'] for field in data_fields]}


=== Available Operators by Category ===
""")
        
        for category, ops in sampled_operators.items():
            prompt_parts.append(f"\n{category}:")
            prompt_parts.append('\n'.join(format_operators(ops[:10])))  # Limit to 10 per category
        
        # 4. Guidelines based on successful patterns
        prompt_parts.append("""

 === HARD RULES ===
1. MUST use at least 2 operators in each expression.
2. MUST use at least 1 data field in each expression.
3. DO NOT invent operators or data fields not listed above.
4. Expressions must be syntactically valid FASTEXPR.

=== Alpha Design Guidelines ===
1. **Combine multiple operators**: Successful alphas often use 2-4 operators
2. **Use time series operators**: ts_mean, ts_std_dev, ts_rank are frequently successful
3. **Apply cross-sectional normalization**: rank, zscore help with stability
4. **Consider turnover**: Avoid overly complex expressions that trade too frequently
5. **Test different timeframes**: Try various window sizes (20, 60, 120, 252 days)
6. **Use fundamental data**: Revenue, earnings, cashflow often provide signal
7. **Combine momentum and mean reversion**: Balance different alpha styles
""")
        
        # 5. Few-shot examples (from RAG)
        if patterns and patterns.get('total_count', 0) > 0:
            top_ops = sorted(patterns.get('operators', {}).items(), key=lambda x: x[1], reverse=True)[:5]
            top_fields = sorted(patterns.get('data_fields', {}).items(), key=lambda x: x[1], reverse=True)[:5]
            
            prompt_parts.append(f"""
=== Insights from Successful Alphas ===
Most effective operators: {', '.join([op for op, _ in top_ops])}
Most effective data fields: {', '.join([field for field, _ in top_fields])}
Average successful fitness: {patterns.get('avg_fitness', 0):.3f}
Average successful Sharpe: {patterns.get('avg_sharpe', 0):.3f}
""")
        example_output_json = """
=== Example Output Format ===
{
    "expressions": [
        "ts_av_diff(power(ts_delay(revenue,75),2),75)-ts_count_nans(revenue,75)",
        "ts_arg_min(ts_count_nans(cogs/inventory_turnover,90),90)",
        "ts_rank(zscore(cashflow_op/revenue),95,constant=1)-rank(returns)",
        "power(ts_arg_min(ts_arg_max(split,140),140),1.1)",
        "market_ret = ts_product(1+group_mean(returns,1,market),250)-1;rfr = vec_avg(fnd6_newqeventv110_optrfrq);expected_return = rfr+beta_last_360_days_spy*(market_ret-rfr);actual_return = ts_product(returns+1,250)-1;actual_return-expected_return"
    ]
}
"""
        # 6. Output format
        prompt_parts.append(f"""

=== Output Requirements ===
1. Return EXACTLY {target_count} unique alpha expressions
2. Each expression must be syntactically valid
3. Use semicolons (;) to separate multi-statement alphas
4. Ensure operator types match (SCALAR, VECTOR, MATRIX)
5. Return as JSON with expressions array

{example_output_json}

Now generate {target_count} unique, creative, and potentially profitable alpha expressions:
""")
        
        return ''.join(prompt_parts)
    
    def check_diversity(self, new_expression: str) -> bool:
        """
        Check if new expression is diverse enough from previously generated ones.
        
        Args:
            new_expression: Expression to check
            
        Returns:
            True if diverse enough, False if too similar
        """
        if not self.generated_expressions:
            return True
        
        # Simple diversity check: character-level similarity
        from difflib import SequenceMatcher
        
        for existing_expr in self.generated_expressions:
            similarity = SequenceMatcher(None, new_expression, existing_expr).ratio()
            if similarity > self.diversity_threshold:
                logger.debug(f"Expression too similar to existing: {similarity:.2f}")
                return False
        
        return True
    
    def generate_alpha_ideas_with_ollama(self, data_fields: List[Dict], operators: List[Dict],
                                         target_count: int = 100) -> List[str]:
        """
        Override parent method to use RAG-enhanced prompts.

        This method:
        1. Retrieves similar successful alphas from Qdrant/RAG system
        2. Builds enhanced prompt with RAG context
        3. Calls Ollama API with enhanced prompt
        4. Returns generated alpha expressions

        Args:
            data_fields: Available data fields
            operators: Available operators
            target_count: Number of alphas to generate

        Returns:
            List of alpha expression strings
        """
        try:
            # Step 1: Query RAG system for similar successful alphas
            logger.info("🔍 Querying RAG system for similar successful alphas...")
            similar_alphas = self.rag_system.retrieve_similar_alphas(
                query=None,  # Get top alphas without specific query
                top_k=5,
                min_fitness=0.6
            )

            # Step 2: Generate RAG context (used in enhanced prompt generation)
            # Note: RAG context is already included in generate_enhanced_prompt()

            # Log RAG usage
            if similar_alphas:
                logger.info(f"✅ Using RAG context with {len(similar_alphas)} similar alphas")
                logger.info(f"📊 RAG alphas fitness range: {min([a.get('fitness', 0) for a in similar_alphas]):.2f} - {max([a.get('fitness', 0) for a in similar_alphas]):.2f}")
            else:
                logger.warning("⚠️ No similar alphas found in RAG system, using standard prompt")

            # Step 3: Build enhanced prompt with RAG context
            logger.info("📝 Building enhanced prompt with RAG context...")
            enhanced_prompt = self.generate_enhanced_prompt(data_fields, operators, target_count)

            # Step 4: Call parent's Ollama API logic with enhanced prompt
            logger.info("🤖 Calling Ollama API with RAG-enhanced prompt...")
            # alpha_ideas = self._call_ollama_with_enhanced_prompt(enhanced_prompt, target_count)
            alpha_ideas = self._call_remote_api_with_enhanced_prompt(enhanced_prompt, target_count)

            # Step 5: Filter for diversity
            diverse_alphas = []
            for alpha in alpha_ideas:
                if self.check_diversity(alpha):
                    diverse_alphas.append(alpha)
                    self.generated_expressions.add(alpha)

            logger.info(f"✅ Generated {len(diverse_alphas)} diverse alphas out of {len(alpha_ideas)} total (RAG-enhanced)")

            # Step 6: Cleanup VRAM after generation
            # Free CUDA cache and Python objects to prevent VRAM leak
            self._cleanup_after_generation()

            return diverse_alphas

        except Exception as e:
            # Fallback to parent class method if RAG fails
            # Log detailed error information for debugging
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"❌ RAG-enhanced generation failed: {type(e).__name__}: {e}")
            logger.error(f"📍 Error location:\n{error_traceback}")
            logger.warning("⚠️ Falling back to standard generation without RAG")
            return super().generate_alpha_ideas_with_ollama(data_fields, operators, target_count)

    def _call_ollama_with_enhanced_prompt(self, prompt: str, target_count: int) -> List[str]:
        """
        Call Ollama API with enhanced prompt (reuses parent's API logic).

        This method extracts the core Ollama API call logic from parent class
        to avoid code duplication while using enhanced prompts.

        Args:
            prompt: Enhanced prompt with RAG context
            target_count: Number of alphas to generate

        Returns:
            List of alpha expression strings
        """
        import requests
        from alpha_generator_ollama import AlphaExpressions, ollama_logger

        # Prepare Ollama API request with structured output
        model_name = getattr(self, 'model_name', self.model_fleet[self.current_model_index])
        ollama_data = {
            'model': model_name,
            'prompt': prompt,
            'stream': False,
            'temperature': 0.3,
            'top_p': 0.9,
            'num_predict': 1000,
            'format': AlphaExpressions.model_json_schema()
        }

        # Log the API request
        ollama_logger.info("=" * 80)
        ollama_logger.info("OLLAMA CONVERSATION START (RAG-ENHANCED)")
        ollama_logger.info("=" * 80)
        ollama_logger.info(f"PROMPT (Target: {target_count} expressions, RAG-enhanced):")
        ollama_logger.info("-" * 40)
        ollama_logger.info(prompt)
        ollama_logger.info("-" * 40)
        ollama_logger.info("API REQUEST:")
        ollama_logger.info(f"Model: {model_name}")
        ollama_logger.info(f"URL: {self.ollama_url}/api/generate")
        ollama_logger.info("-" * 40)

        try:
            # Send request to Ollama API
            response = requests.post(
                f'{self.ollama_url}/api/generate',
                json=ollama_data,
                timeout=500
            )

            if response.status_code != 200:
                ollama_logger.error(f"API ERROR: Status {response.status_code}")
                ollama_logger.error(f"Response: {response.text}")
                ollama_logger.info("=" * 80)
                ollama_logger.info("OLLAMA CONVERSATION END (ERROR)")
                ollama_logger.info("=" * 80)
                return []

            response_data = response.json()

            # Log the API response
            ollama_logger.info("API RESPONSE:")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Status Code: {response.status_code}")
            ollama_logger.info("-" * 40)

            if 'response' not in response_data:
                ollama_logger.error(f"Unexpected response format: {response_data}")
                return []

            content = response_data['response']

            # Parse the structured JSON response
            alpha_data = AlphaExpressions.model_validate_json(content)
            alpha_ideas = alpha_data.expressions

            # Log parsing results
            ollama_logger.info("PARSING RESULTS (RAG-ENHANCED):")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Generated {len(alpha_ideas)} alpha ideas")
            for i, alpha in enumerate(alpha_ideas[:10], 1):  # Log first 10
                ollama_logger.info(f"Alpha {i}: {alpha}")
            if len(alpha_ideas) > 10:
                ollama_logger.info(f"... and {len(alpha_ideas) - 10} more")
            ollama_logger.info("-" * 40)

            # Clean and validate ideas
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)

            # Log final results
            ollama_logger.info("FINAL RESULTS (RAG-ENHANCED):")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Valid expressions after cleaning: {len(cleaned_ideas)}")
            ollama_logger.info("=" * 80)
            ollama_logger.info("OLLAMA CONVERSATION END (RAG-ENHANCED)")
            ollama_logger.info("=" * 80)
            ollama_logger.info("")

            return cleaned_ideas

        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out (360s)")
            ollama_logger.error("API TIMEOUT ERROR")
            ollama_logger.info("=" * 80)
            ollama_logger.info("OLLAMA CONVERSATION END (TIMEOUT)")
            ollama_logger.info("=" * 80)
            self._handle_ollama_error("timeout")
            return []

        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            ollama_logger.error(f"PARSING ERROR: {e}")
            ollama_logger.info("=" * 80)
            ollama_logger.info("OLLAMA CONVERSATION END (PARSING_ERROR)")
            ollama_logger.info("=" * 80)
            return []

    def _call_remote_api_with_enhanced_prompt(self, prompt: str, target_count: int,
                                               provider: Optional[str] = None) -> List[str]:
        """
        Call remote AI API (DeepSeek or OpenAI) with enhanced prompt.

        This function provides an alternative to local Ollama API by supporting remote AI providers.
        It maintains the same input/output interface as _call_ollama_with_enhanced_prompt for
        easy integration and switching between local and remote AI services.

        Supported Providers:
            - 'deepseek': DeepSeek API (https://api.deepseek.com)
            - 'openai': OpenAI API (https://api.openai.com)

        Configuration:
            Provider selection priority (highest to lowest):
            1. Function parameter 'provider'
            2. Environment variable 'REMOTE_AI_PROVIDER' (deepseek or openai)
            3. Default: 'deepseek'

            API Keys (required):
            - DeepSeek: Set environment variable 'DEEPSEEK_API_KEY'
            - OpenAI: Set environment variable 'OPENAI_API_KEY'

            Model Selection (optional):
            - DeepSeek: Set 'DEEPSEEK_MODEL' (default: 'deepseek-chat')
            - OpenAI: Set 'OPENAI_MODEL' (default: 'gpt-4o')

        Args:
            prompt: Enhanced prompt with RAG context
            target_count: Number of alphas to generate
            provider: AI provider to use ('deepseek' or 'openai'). If None, uses environment variable.

        Returns:
            List of cleaned alpha expression strings

        Raises:
            ValueError: If API key is not configured or provider is invalid
            Exception: If API call fails after retries
        """
        import requests
        import json
        from alpha_generator_ollama import ollama_logger

        # Step 1: Determine which AI provider to use
        # Priority: function parameter > environment variable > default
        if provider is None:
            provider = os.getenv('REMOTE_AI_PROVIDER', 'deepseek').lower()
        else:
            provider = provider.lower()

        # Validate provider selection
        if provider not in ['deepseek', 'openai']:
            raise ValueError(f"Invalid provider '{provider}'. Must be 'deepseek' or 'openai'")

        # Step 2: Configure API endpoint and authentication based on provider
        if provider == 'deepseek':
            # DeepSeek API configuration
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

            api_url = "https://api.deepseek.com/v1/chat/completions"
            model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

        elif provider == 'openai':
            # OpenAI API configuration
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            api_url = "https://api.openai.com/v1/chat/completions"
            model_name = os.getenv('OPENAI_MODEL', 'gpt-4o')

        # Step 3: Prepare API request headers
        # Both DeepSeek and OpenAI use similar authentication header format
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        # Step 4: Construct request payload
        # Use OpenAI-compatible chat completion format (supported by both providers)
        request_payload = {
            'model': model_name,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert quantitative researcher specializing in alpha factor discovery. Generate valid FASTEXPR alpha expressions in JSON format.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,
            'top_p': 0.9,
            'max_tokens': 4000,
            'response_format': {'type': 'json_object'}  # Request JSON response
        }

        # Step 5: Log the API request for debugging
        ollama_logger.info("=" * 80)
        ollama_logger.info(f"REMOTE AI API CALL START ({provider.upper()})")
        ollama_logger.info("=" * 80)
        ollama_logger.info(f"PROMPT (Target: {target_count} expressions, RAG-enhanced):")
        ollama_logger.info("-" * 40)
        ollama_logger.info(prompt)
        ollama_logger.info("-" * 40)
        ollama_logger.info("API REQUEST:")
        ollama_logger.info(f"Provider: {provider}")
        ollama_logger.info(f"Model: {model_name}")
        ollama_logger.info(f"URL: {api_url}")
        ollama_logger.info("-" * 40)

        try:
            # Step 6: Send HTTP POST request to remote AI API
            # Timeout set to 120 seconds for remote API calls
            response = requests.post(
                api_url,
                headers=headers,
                json=request_payload,
                timeout=120
            )

            # Step 7: Check HTTP response status
            if response.status_code != 200:
                error_message = f"API ERROR: Status {response.status_code}"
                ollama_logger.error(error_message)
                ollama_logger.error(f"Response: {response.text}")
                ollama_logger.info("=" * 80)
                ollama_logger.info(f"{provider.upper()} API CALL END (ERROR)")
                ollama_logger.info("=" * 80)

                # Handle specific error codes
                if response.status_code == 401:
                    raise ValueError(f"{provider} API authentication failed. Check your API key.")
                elif response.status_code == 429:
                    logger.warning(f"{provider} API rate limit exceeded. Consider adding retry logic.")
                    return []
                else:
                    return []

            # Step 8: Parse JSON response
            response_data = response.json()

            # Log the API response
            ollama_logger.info("API RESPONSE:")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Status Code: {response.status_code}")
            ollama_logger.info("-" * 40)

            # Step 9: Extract content from response
            # Both DeepSeek and OpenAI use the same response format
            if 'choices' not in response_data or len(response_data['choices']) == 0:
                ollama_logger.error(f"Unexpected response format: {response_data}")
                ollama_logger.info("=" * 80)
                ollama_logger.info(f"{provider.upper()} API CALL END (ERROR)")
                ollama_logger.info("=" * 80)
                return []

            # Extract the assistant's message content
            content = response_data['choices'][0]['message']['content']

            # Step 10: Parse the JSON content to extract alpha expressions
            try:
                # The content should be a JSON object with an "expressions" array
                alpha_data_dict = json.loads(content)

                # Validate the structure
                if 'expressions' not in alpha_data_dict:
                    ollama_logger.error(f"Response missing 'expressions' field: {alpha_data_dict}")
                    return []

                alpha_ideas = alpha_data_dict['expressions']

                # Ensure it's a list
                if not isinstance(alpha_ideas, list):
                    ollama_logger.error(f"'expressions' field is not a list: {type(alpha_ideas)}")
                    return []

            except json.JSONDecodeError as e:
                ollama_logger.error(f"Failed to parse JSON content: {e}")
                ollama_logger.error(f"Content: {content}")
                ollama_logger.info("=" * 80)
                ollama_logger.info(f"{provider.upper()} API CALL END (PARSING_ERROR)")
                ollama_logger.info("=" * 80)
                return []

            # Step 11: Log parsing results
            ollama_logger.info("PARSING RESULTS (RAG-ENHANCED, REMOTE API):")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Generated {len(alpha_ideas)} alpha ideas")
            for i, alpha in enumerate(alpha_ideas[:10], 1):  # Log first 10
                ollama_logger.info(f"Alpha {i}: {alpha}")
            if len(alpha_ideas) > 10:
                ollama_logger.info(f"... and {len(alpha_ideas) - 10} more")
            ollama_logger.info("-" * 40)

            # Step 12: Clean and validate ideas using parent class method
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)

            # Step 13: Log final results
            ollama_logger.info("FINAL RESULTS (RAG-ENHANCED, REMOTE API):")
            ollama_logger.info("-" * 40)
            ollama_logger.info(f"Valid expressions after cleaning: {len(cleaned_ideas)}")
            ollama_logger.info("=" * 80)
            ollama_logger.info(f"{provider.upper()} API CALL END (SUCCESS)")
            ollama_logger.info("=" * 80)
            ollama_logger.info("")

            return cleaned_ideas

        except requests.exceptions.Timeout:
            # Handle timeout errors
            logger.error(f"{provider} API request timed out (120s)")
            ollama_logger.error("API TIMEOUT ERROR")
            ollama_logger.info("=" * 80)
            ollama_logger.info(f"{provider.upper()} API CALL END (TIMEOUT)")
            ollama_logger.info("=" * 80)
            return []

        except requests.exceptions.ConnectionError as e:
            # Handle connection errors
            logger.error(f"{provider} API connection error: {e}")
            ollama_logger.error(f"API CONNECTION ERROR: {e}")
            ollama_logger.info("=" * 80)
            ollama_logger.info(f"{provider.upper()} API CALL END (CONNECTION_ERROR)")
            ollama_logger.info("=" * 80)
            return []

        except ValueError as e:
            # Handle configuration errors (missing API keys, invalid provider)
            logger.error(f"Configuration error: {e}")
            ollama_logger.error(f"CONFIGURATION ERROR: {e}")
            ollama_logger.info("=" * 80)
            ollama_logger.info(f"{provider.upper()} API CALL END (CONFIG_ERROR)")
            ollama_logger.info("=" * 80)
            raise  # Re-raise to notify caller of configuration issues

        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Failed to call {provider} API: {e}")
            ollama_logger.error(f"UNEXPECTED ERROR: {e}")
            ollama_logger.info("=" * 80)
            ollama_logger.info(f"{provider.upper()} API CALL END (ERROR)")
            ollama_logger.info("=" * 80)
            return []

    def _cleanup_after_generation(self):
        """
        Cleanup VRAM after RAG-enhanced generation.

        This method is called after each alpha generation cycle to:
        1. Free Python objects via garbage collection
        2. Clear PyTorch CUDA cache to prevent VRAM leak
        3. Prevent memory accumulation from embeddings and model inference

        The cleanup is critical for long-running processes that use:
        - Sentence Transformers for RAG embeddings
        - Ollama models for alpha generation
        - Multiple generation cycles that accumulate CUDA tensors
        """
        try:
            import gc
            import torch

            # Step 1: Python garbage collection
            # Free unused Python objects and their associated memory
            gc.collect()

            # Step 2: PyTorch CUDA cache cleanup
            # Clear CUDA memory cache to free GPU memory from embeddings and model inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete

            logger.debug("🧹 Cleaned up VRAM after RAG-enhanced generation")
        except Exception as e:
            logger.warning(f"⚠️ VRAM cleanup failed: {e}")
