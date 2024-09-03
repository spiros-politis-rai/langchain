import json
import logging
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.utils import guard_import

from langchain_community.utilities.snowflake import SnowflakeConnector

# Check for `snowflake-snowpark-python` package.
guard_import("snowflake.snowpark", pip_name="snowflake-snowpark-python")

logger = logging.getLogger(__name__)


class SnowflakeCortexSQL(LLM):
    """Wraps calling Snowflake Cortex LLMs with the SQL interface."""

    """The Snowflake connector class to use."""
    connector: SnowflakeConnector = None

    """The Snowflake Cortex model to call.
    
    Default: "mistral-7b"
    """
    model: str = "mistral-7b"

    """A value from 0 to 1 (inclusive) that controls the randomness 
    of the output of the language model. 
    A higher temperature (for example, 0.7) results in more diverse 
    and random output, while a lower temperature (such as 0.2) makes 
    the output more deterministic and focused.

    Default: 0.0
    """
    temperature: Optional[float] = 0.0

    """A value from 0 to 1 (inclusive) that controls the randomness and diversity 
    of the language model, generally used as an alternative to temperature. 
    The difference is that top_p restricts the set of possible tokens that 
    the model outputs, while temperature influences which tokens 
    are chosen at each step.
    
    Default: 0.0
    """
    top_p: Optional[float] = 0.0

    """Sets the maximum number of output tokens in the response. 
    Small values can result in truncated responses.

    Default: 4096
    """
    max_tokens: Optional[int] = 4096

    """Filters potentially unsafe and harmful responses from a language model. 
    Either true or false.

    Default: False
    """
    guardrails: Optional[bool] = False

    SUPPORTED_MODELS = [
        "snowflake-arctic",
        "mistral-large",
        "mistral-large2",
        "reka-flash",
        "reka-core",
        "mixtral-8x7b",
        "jamba-instruct",
        "llama2-70b-chat",
        "llama3-8b",
        "llama3-70b",
        "llama3.1-8b",
        "llama3.1-70b",
        "llama3.1-405b",
        "mistral-7b",
        "gemma-7b",
    ]

    def _validate_model(self, value):
        if value.lower() not in [model.lower() for model in self.SUPPORTED_MODELS]:
            raise ValueError(
                f"Model {value} is not supported. Please choose from \
                    {self.SUPPORTED_MODELS}"
            )
        return value

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call a Snowflake Cortex model.

        When the `options` argument is not specified, the call returns a string.

        When the `options` argument is given, the call returns a string
        representation of a JSON object containing the following keys:

            - `choices`: An array of the model’s responses.
            (Currently, only one response is provided.)
            Each response is an object containing a "messages" key
            whose value is the model’s response to the latest prompt.

            - `created`: UNIX timestamp (seconds since midnight, January 1, 1970)
            when the response was generated.

            - `model`: The name of the model that created the response.

            - `usage`: An object recording the number of tokens consumed and generated
            by this completion. Includes the following sub-keys:

            - `completion_tokens`: The number of tokens in the generated response.

            - `prompt_tokens`: The number of tokens in the prompt.

            - `total_tokens`: The total number of tokens consumed,
            which is the sum of the other two values.

        Example:
            .. code-block:: python

                from langchain_community.llms.snowflake import SnowflakeCortex

                llm = SnowflakeCortexSQL(
                    model="mistral-7b",
                    temperature=0.0,
                    top_p=0.0,
                    guardrails=False
                )
                llm.connector = snowflake_connector_basic

                result = llm.invoke(
                    "What is the first letter in the Greek Alphabet?"
                )

        Args:
            prompt: The prompt to complete.
            stop: The stop tokens.
            run_manager: The run manager.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The LLM completion.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not allowed.")

        self._validate_model(self.model)

        params = {**self._default_params, **kwargs}
        query = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{self.model}', 
                [
                    {{
                        'role': 'user', 
                        'content': '{prompt}'
                    }}
                ],
                {params} 
            ) AS completion;
        """

        try:
            with self.connector.connect() as session:
                return json.loads(
                    session.connection.cursor().execute(query).fetchall()[0][0]
                )["choices"][0]["messages"]
        except Exception as error:
            logger.error(f"Error calling Snowflake Cortex model {self.model}")
            logger.error(error)
            raise error

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling AI21 API."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "guardrails": self.guardrails,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return a dictionary of the identifying parameters"""
        return {"connector": self.connector, "model": self.model}

    @property
    def _llm_type(self) -> str:
        return "snowflake-cortex"
