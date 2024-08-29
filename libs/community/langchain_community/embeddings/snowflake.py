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

    """Snowflake embeddings model to use."""
    model: str = "e5-base-v2"

    """Show progress bar. Requires ``tqdm`` to be installed."""
    show_progress: bool = False

    embeddings_dim: int = 768

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        fields = {
            "connector": "connector", 
            "model": "model", 
            "embeddings_dim": "embeddings_dim", 
            "show_progress": "show_progress"
        }

    def _get_session(self):
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
                "show_progress": self.show_progress
            },
            **self._default_params,
        }

    def _get_snowflake_embeddings(self, input: str) -> List[float]:
        """Process response from Snowflake.

        Args:
            response: The response from Snowflake.

        Returns:
            The response as a dictionary.
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
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="SnowflakeEmbeddings")
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported."
                    "Please install with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._get_snowflake_embeddings(item) for item in iter_]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Embed documents using Snowflake's embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            items = [item for item in input]
            return self._get_embeddings(items)
        except Exception as error:
            logger.error("Error producing Snowflake embeddings")
            logger.error(error)
            raise error

    def embed_query(self, input: str) -> List[float]:
        """Produce text embeddings using a Snowflake embeddings model.

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
