"""
Common utilities for Gemini API interactions.
"""

import logging
from typing import Optional, Dict, Any, Type

import google.generativeai as genai
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


logger = logging.getLogger("gemini_update")


class CommonGeminiTools:
    """Provides common utilities for interacting with the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, enable_search_grounding: bool = True):
        """
        Initialize CommonGeminiTools.

        Args:
            api_key: Optional Gemini API key. If provided, configures the API.
            enable_search_grounding: Flag to enable/disable search grounding features.
        """
        self.enable_search_grounding = enable_search_grounding
        if api_key:
            self.configure_api(api_key)

    def configure_api(self, api_key: str) -> bool:
        """
        Configure the Google Gemini API with the provided API key and test the connection.

        Args:
            api_key: The Gemini API key.

        Returns:
            True if configuration and connection test are successful, False otherwise.
        """
        if not api_key:
            logger.error("Gemini API key not provided for configuration.")
            return False
        try:
            genai.configure(api_key=api_key)
            self.get_available_model()
            logger.info("Gemini API configured and connection successful.")
            return True
        except Exception as e:
            logger.error(f"Gemini API configuration or connection test failed: {str(e)}")
            return False

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieve details and capabilities for a given Gemini model name.
        Provides conservative defaults if the model is not found or details are missing.

        Args:
            model_name: The name of the model (e.g., "gemini-1.5-pro").

        Returns:
            A dictionary containing model parameters like temperature, top_p, top_k,
            max_output_tokens, and candidate_count.
        """
        try:
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            models = genai.list_models()
            model_info = None
            for model_obj in models:
                if model_obj.name.endswith(model_name):
                    model_info = model_obj
                    logger.info(f"Found model details for {model_name}")
                    break
            if model_info is None:
                logger.warning(f"Could not find details for model {model_name}, using conservative defaults")
                return {"temperature": 0.2, "top_p": 0.95, "top_k": 32, "max_output_tokens": 4096, "candidate_count": 1}
            
            output_token_limit = getattr(model_info, 'output_token_limit', 4096)
            default_temp = getattr(model_info, 'temperature', 1.0)
            max_temp = getattr(model_info, 'max_temperature', 2.0)
            if max_temp is None: max_temp = 2.0
            default_top_p = getattr(model_info, 'top_p', 0.95)
            default_top_k = getattr(model_info, 'top_k', 40)
            
            logger.info(f"Model limits: output_tokens={output_token_limit}, max_temp={max_temp}, top_p={default_top_p}, top_k={default_top_k}")
            
            return {
                "temperature": min(0.2, max_temp if max_temp is not None else 0.2),
                "top_p": default_top_p,
                "top_k": default_top_k,
                "max_output_tokens": min(16384, output_token_limit),
                "candidate_count": 1,
                "max_temperature": max_temp
            }
        except Exception as e:
            logger.warning(f"Error getting model details for {model_name}: {str(e)}, using conservative defaults")
            return {"temperature": 0.2, "top_p": 0.95, "top_k": 32, "max_output_tokens": 4096, "candidate_count": 1}

    def get_available_model(self, requested_model: str = 'gemini-1.5-pro') -> str:
        """
        Find an available Gemini model, trying the requested model first,
        then falling back to other available models based on a priority.

        Args:
            requested_model: The preferred model name.

        Returns:
            The name of a working, available Gemini model.
        """
        try:
            logger.info(f"Trying requested model: {requested_model}")
            model_to_test = genai.GenerativeModel(requested_model)
            model_to_test.generate_content("Test")
            logger.info(f"Successfully using requested model: {requested_model}")
            return requested_model
        except Exception as e:
            logger.warning(f"Requested model {requested_model} not available or failed test: {str(e)}")

        try:
            logger.info("Fetching list of available models from Gemini API...")
            available_models = genai.list_models()
            content_models = [
                model.name.replace("models/", "") 
                for model in available_models 
                if 'generateContent' in getattr(model, 'supported_generation_methods', [])
            ]
            if not content_models:
                logger.warning("No models supporting generateContent found, falling back to gemini-pro.")
                return "gemini-pro"

            sorted_models = sorted(content_models, key=lambda m: ("pro" not in m, "flash" not in m, m), reverse=False)
            
            logger.info(f"Available models sorted (simplified): {sorted_models[:5]} (showing top 5)")
            for candidate in sorted_models:
                try:
                    logger.info(f"Trying candidate model: {candidate}")
                    model_to_test = genai.GenerativeModel(candidate)
                    model_to_test.generate_content("Test")
                    logger.info(f"Successfully found working model: {candidate}")
                    return candidate
                except Exception as e_candidate:
                    logger.warning(f"Candidate model {candidate} failed: {str(e_candidate)}")
                    continue
        except Exception as e_list:
            logger.error(f"Error getting available models: {str(e_list)}")
        
        logger.warning("All model attempts failed, using legacy gemini-pro as final attempt.")
        return "gemini-pro"

    def _validate_token_config(self, token_config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Validate and adjust token configuration parameters against model capabilities.

        Args:
            token_config: The user-provided token configuration.
            model_name: The name of the model to validate against.

        Returns:
            A dictionary with validated token configurations.
        """
        model_capabilities = self.get_model_details(model_name)
        validated_config = token_config.copy()

        max_temp = model_capabilities.get("max_temperature", 2.0)
        if "temperature" in validated_config:
            if validated_config["temperature"] > max_temp:
                logger.warning(f"Temperature {validated_config['temperature']} exceeds model maximum {max_temp}, adjusting to {max_temp}")
                validated_config["temperature"] = max_temp
            elif validated_config["temperature"] < 0:
                 logger.warning(f"Temperature {validated_config['temperature']} is below 0, adjusting to 0")
                 validated_config["temperature"] = 0.0

        model_max_tokens = model_capabilities.get("max_output_tokens", 4096)
        if "max_output_tokens" in validated_config:
            if validated_config["max_output_tokens"] > model_max_tokens:
                logger.warning(f"max_output_tokens {validated_config['max_output_tokens']} exceeds model limit {model_max_tokens}, adjusting.")
                validated_config["max_output_tokens"] = model_max_tokens
            elif validated_config["max_output_tokens"] <=0:
                logger.warning(f"max_output_tokens {validated_config['max_output_tokens']} must be positive, adjusting to {model_max_tokens}.")
                validated_config["max_output_tokens"] = model_max_tokens

        model_top_p = model_capabilities.get("top_p", 0.95)
        if "top_p" in validated_config:
            if not (0 <= validated_config["top_p"] <= 1):
                logger.warning(f"top_p {validated_config['top_p']} is outside valid range [0,1], adjusting to model default {model_top_p}")
                validated_config["top_p"] = model_top_p
        
        model_top_k = model_capabilities.get("top_k", 40)
        if "top_k" in validated_config:
            if validated_config["top_k"] <= 0:
                logger.warning(f"top_k {validated_config['top_k']} must be positive, adjusting to model default {model_top_k}")
                validated_config["top_k"] = model_top_k
        
        return validated_config

    def create_pydantic_agent(self, model_name: str, token_config: Dict,
                          deps_type: Type[BaseModel], output_type: Type[BaseModel],
                          system_prompt_str: str) -> Agent:
        """
        Create a pydantic-ai Agent configured with a Gemini model.

        Args:
            model_name: The name of the Gemini model to use.
            token_config: Dictionary of token settings (temperature, top_p, etc.).
            deps_type: The Pydantic model type for agent dependencies (input).
            output_type: The Pydantic model type for the agent's structured output.
            system_prompt_str: The base system prompt string for the agent.

        Returns:
            A configured pydantic-ai Agent instance.
        """
        working_model = self.get_available_model(model_name)
        logger.info(f"Creating agent with model: {working_model}")

        validated_token_config = self._validate_token_config(token_config, working_model)

        agent_kwargs = {
            "model": working_model,
            "deps_type": deps_type,
            "output_type": output_type,
            "system_prompt": system_prompt_str,
            "settings": validated_token_config
        }

        logger.info(f"Creating pydantic-ai agent with model: {working_model}, system prompt length: {len(system_prompt_str)}")
        return Agent(**agent_kwargs)
