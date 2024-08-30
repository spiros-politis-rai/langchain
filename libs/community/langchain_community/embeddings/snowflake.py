import logging
from typing import Any, List, Mapping

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

try:
    from snowflake.snowpark import Session
except ImportError:
    raise ImportError(
        "`snowflake-snowpark-python` package not found, please install it with "
        "`pip install snowflake-snowpark-python`"
    )

from langchain_community.utilities.snowflake import SnowflakeConnector

logger = logging.getLogger(__name__)


class SnowflakeEmbeddings(BaseModel, Embeddings):
    """Snowflake embeddings.

    Example:
        .. code-block:: python

            from langchain_community.embeddings.snowflake import SnowflakeEmbeddings
            snowflake_embeddings = SnowflakeEmbeddings(
                model="e5-base-v2",
            )
            e_1 = snowflake_embeddings.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            e_2 = snowflake_embeddings.embed_query(
                "What is the second letter of Greek alphabet"
            )
    """

    """Snowflake connector class to use.

    This must be set after instantiation of the SnowflakeEmbeddings class.
    """
    connector: SnowflakeConnector = None

    """Snowflake embeddings model to use.
    
    Currently, available Snowflake embeddings models are the following:
        - `e5-base-v2`: produces 768-dimensional embeddings
        - `nv-embed-qa-4`: produces 1024-dimensional embeddings

    Default is `e5-base-v2`.
    """
    model: str = "e5-base-v2"

    """Show progress bar. Requires ``tqdm`` to be installed."""
    show_progress: bool = False

    """Embeddings dimensionality."""
    embeddings_dim: int = 768

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        fields = {
            "connector": "connector",
            "model": "model",
            "embeddings_dim": "embeddings_dim",
            "show_progress": "show_progress",
        }

    def _get_session(self) -> Session:
        """Get a Snowflake session.

        Returns:
            A Snoflake Session object.
        """
        try:
            return self.connector.connect()
        except Exception as error:
            logger.error("Error connecting to Snowflake")
            logger.error(error)
            raise error

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{
                "model": self.model,
                "connector": self.connector,
                "embeddings_dim": self.embeddings_dim,
                "show_progress": self.show_progress,
            },
            **self._default_params,
        }

    def _get_snowflake_embeddings(self, input: str) -> List[float]:
        """Call the Snowflake SQL API to retrieve embeddings.

        Args:
            input: The string for which to retrieve embeddings from Snowflake.

        Returns:
            A list of floats representing the embeddings.
        """
        # Basic safeguards against model / dim mismatch.
        if self.model in ["e5-base-v2"] and self.embeddings_dim != 768:
            raise ValueError(
                "Only e5-base-v2 model is supported for 768-dim embeddings"
            )
        elif self.model in ["nv-embed-qa-4"] and self.embeddings_dim != 1024:
            raise ValueError(
                "Only nv-embed-qa-4 model is supported for 1024-dim embeddings"
            )

        escaped_input = input.replace("'", "\\'")
        query = f"""
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_{self.embeddings_dim}
            ('{self.model}', '{escaped_input}') 
            AS embeddings
        """
        result = None
        try:
            with self._get_session() as session:
                result = session.connection.cursor().execute(query).fetchone()[0]
        except Exception as error:
            logger.error("Error getting Snowflake embeddings")
            logger.error(error)
            raise error
        return result

    def _get_embeddings(self, input: List[str]) -> List[List[float]]:
        """Pass-through function to get embeddings for a list of strings,
        utilizing the progress bar if requested and TQDM can be imported.
        """
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="SnowflakeEmbeddings")
            except ImportError:
                logger.warning(
                    "Cannot show progress bar as `tqdm` could not be imported."
                    "Please install tqdm with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._get_snowflake_embeddings(item) for item in iter_]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Produce embeddings for a list of strings, using Snowflake's embedding model.

        Example:
            .. code-block:: python

                embeddings = snowflake_embeddings.embed_documents(
                    [
                        "Alpha is the first letter of Greek alphabet",
                        "Beta is the second letter of Greek alphabet"
                    ]
                )

        Args:
            input: The list of texts to retrieve embeddings for.

        Returns:
            List of lists of float embeddings, one for each text.
        """
        try:
            items = [item for item in input]
            return self._get_embeddings(items)
        except Exception as error:
            logger.error("Error producing Snowflake embeddings")
            logger.error(error)
            raise error

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of lists of floats as embeddings of the input.
        """
        raise NotImplementedError("Not implemented")

    def embed_query(self, input: str) -> List[float]:
        """Produce embeddings for a string, using Snowflake's embedding model.

        Example:
            .. code-block:: python

                embeddings = snowflake_embeddings.embed_query(
                    "Alpha is the first letter of Greek alphabet"
                )

        Args:
            input: the text to embed.

        Returns:
            Embeddings of the input.
        """
        try:
            return self._get_embeddings([input])[0]
        except Exception as error:
            logger.error("Error producing Snowflake embeddings")
            logger.error(error)
            raise error

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: The text to embed.

        Returns:
            List of floats as embeddings of the input.
        """
        raise NotImplementedError("Not implemented")
