from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VST, VectorStoreRetriever

from langchain_community.utilities.snowflake import SnowflakeConnector

# Check for `snowflake-snowpark-python` package.
guard_import("snowflake.snowpark", pip_name="snowflake-snowpark-python")

logger = logging.getLogger(__name__)


class SnowflakeVectorStore(VectorStore):
    """Wrapper around Snowflake vector data type used as vector store."""

    _DEFAULT_SNOWFLAKE_TABLE_NAME: str = "langchain"
    _DEFAULT_EMBEDDINGS_DIM = 768

    def __init__(
        self,
        connector: SnowflakeConnector,
        embeddings: Embeddings,
        embeddings_dim: int = _DEFAULT_EMBEDDINGS_DIM,
        table: str = _DEFAULT_SNOWFLAKE_TABLE_NAME,
    ):
        self._connector = connector
        self._table = table
        self._embeddings = embeddings
        self._embeddings_dim = embeddings_dim

        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create the Snowflake table to persist data."""
        with self._connector.connect() as session:
            try:
                session.connection.cursor().execute(
                    f"""
                        CREATE TABLE IF NOT EXISTS {self._table} (
                            id INTEGER AUTOINCREMENT START 1 INCREMENT 1 ORDER, 
                            hash VARCHAR, 
                            text VARCHAR, 
                            metadata VARIANT, 
                            embeddings VECTOR(FLOAT, {self._embeddings_dim})
                        );
                    """
                )
            except Exception as error:
                logger.error(f"Error creating Snowflake table `{self._table}`")
                logger.error(error)
                raise error

    def _get_topk_similar_with_score(
        self, embeddings: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        result: List[Tuple[Document, float]] = []

        query = f"""
            WITH search_embeddings AS ( 
                SELECT {embeddings}::VECTOR(float, {self._embeddings_dim}) 
                AS embeddings 
            ) 
            SELECT 
                t.text, 
                t.metadata, 
                VECTOR_COSINE_SIMILARITY(t.embeddings, s.embeddings) AS score 
            FROM 
                {self._table} t, 
                search_embeddings s 
            ORDER BY 
                score DESC 
            LIMIT {k} 
        """

        try:
            with self._connector.connect() as session:
                result_set = session.connection.cursor().execute(query).fetchall()

            for row in result_set:
                text = row[0]
                metadata = json.loads(row[1]) or {}
                score = row[2]
                result.append((Document(page_content=text, metadata=metadata), score))
            return result
        except Exception as error:
            logger.error("Error retrieving cosine-similar embeddings from Snowflake")
            logger.error(error)
            raise error

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore index.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.add_texts(
                    texts=[
                        "Alpha is the first letter of Greek alphabet",
                        "Beta is the second letter of Greek alphabet"
                    ],
                    metadatas=[
                        {"some_key_1": "some value"},
                        {"some_key_2": "some value"}
                    ]
                )

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadata associated with the content items.
            kwargs: Vector store specific parameters.

        Returns:
            List of IDs of the added texts.
        """
        # Retrieve current max ID from the table.
        try:
            with self._connector.connect() as session:
                max_id = (
                    session.connection.cursor()
                    .execute(
                        f"""
                            SELECT NVL(MAX(id), 0) AS id 
                            FROM {self._table}
                        """
                    )
                    .fetchone()[0]
                )
        except Exception as error:
            logger.error(
                f"Error retrieving max ID from Snowflake table `{self._table}`"
            )
            logger.error(error)
            raise error

        # Initialize empty metadata dicts if metadata is not provided.
        if not metadatas:
            metadatas = [{} for _ in texts]

        data = [
            (text_item, json.dumps(metadata), embeddings)
            for text_item, metadata, embeddings in zip(
                texts, metadatas, self._embeddings.embed_documents(list(texts))
            )
        ]

        # Merge data into Snowflake table.
        try:
            with self._connector.connect() as session:
                session.connection.cursor().execute("begin")
                for row in data:
                    hash = hashlib.sha256(row[0].encode("UTF-8")).hexdigest()
                    text = row[0].replace("'", "\\'")
                    metadata = row[1]
                    embeddings = row[2]
                    merge_data_query = f"""
                        MERGE INTO {self._table} e USING (
                        SELECT
                            '{hash}'::VARCHAR AS hash, 
                            '{text}'::VARCHAR AS text, 
                            PARSE_JSON('{metadata}') AS metadata, 
                            {embeddings}::VECTOR(float, {self._embeddings_dim}) 
                            AS embeddings 
                        ) AS n 
                        ON e.hash = n.hash 
                        WHEN NOT MATCHED THEN 
                            INSERT (hash, text, metadata, embeddings) 
                            VALUES (n.hash, n.text, n.metadata, n.embeddings)
                        WHEN MATCHED AND e.metadata <> n.metadata THEN
                            UPDATE SET metadata = n.metadata 
                    """
                    session.connection.cursor().execute(merge_data_query)
                session.connection.cursor().execute("commit")
        except Exception as error:
            logger.error(f"Error adding data to Snowflake table `{self._table}`")
            logger.error(error)
            raise error

        # Retrieve last inserted IDs.
        retrieve_data_query = f"""
            SELECT id 
            FROM {self._table} 
            WHERE id > {max_id}
        """

        try:
            with self._connector.connect() as session:
                results = session.connection.cursor().execute(retrieve_data_query)
            return [row[0] for row in results]
        except Exception as error:
            logger.error(
                f"Error retrieving last inserted IDs from Snowflake table \
                    `{self._table}`"
            )
            logger.error(error)
            raise error

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError("Method not implemented")

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vectorstore index.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.add_documents(
                    documents=[
                        Document(
                            page_content="Alpha is the first letter of Greek alphabet"
                        ),
                        Document(
                            page_content="Beta is the second letter of Greek alphabet"
                        )
                    ]
                )

        Args:
            documents: Iterable of Document to add to the vector store.
            metadatas: Optional list of metadata associated with the content items.
            kwargs: Vector store specific parameters.

        Returns:
            List of IDs of the added texts.
        """
        try:
            result = self.add_texts(
                texts=[document.page_content for document in documents],
                metadatas=[document.metadata for document in documents],
                **kwargs,
            )
            return result
        except Exception as error:
            logger.error(f"Error adding documents to Snowflake table `{self._table}`")
            logger.error(error)
            raise error

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        raise NotImplementedError("Method not implemented")

    # TODO: ids are optional, delete everything if ids is None.
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete IDs from the vectorstore.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.delete(
                    ids=[1, 2, 3]
                )

        Args:
            ids: List of IDs to delete from the vector store.
            kwargs: Vector store specific parameters.

        Returns:
            True if operation was successful.
        """
        try:
            ids_str = ", ".join(str(v) for v in ids)
            with self._connector.connect() as session:
                session.connection.cursor().execute(
                    f"""
                        DELETE FROM {self._table}
                        WHERE id IN ({ids_str})
                    """
                ).fetchone()[0]
                return True
        except Exception as error:
            logger.error(f"Error deleting from Snowflake table `{self._table}`")
            logger.error(error)
            raise error

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        raise NotImplementedError("Method not implemented")

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        """Return documents most similar to query, using specified search type.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.search(
                    search_type="similarity",
                    query="What is the first letter in the Greek alphabet?",
                    k=3
                )

        Args:
            query: String query to match against.
            search_type: Type of search to perform. Can be `similarity`, `mmr`
            or `similarity_score_threshold`.
            kwargs: Vector store specific parameters.

        Returns:
            List of matching Documents.
        """
        if search_type == "similarity":
            return self.similarity_search(query=query, **kwargs)
        elif search_type == "mmr":
            raise NotImplementedError("search not implemented for MMR algorithm")
        elif search_type == "similarity_score_threshold":
            return self.similarity_search_with_relevance_scores(query=query, **kwargs)
        else:
            raise ValueError(
                "search type may ony be one of \
                'similarity', 'mmr'' or 'similarity_score_threshold'"
            )

        # raise NotImplementedError("Method not implemented")

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError("Method not implemented")

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to query.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.similarity_search(
                    query="What is the first letter in the Greek alphabet?",
                    k=3
                )

        Args:
            query: String query to match against.
            k: Number of matching Documents to return. Defaults to 4.
            kwargs: Vector store specific parameters.

        Returns:
            List of matching Documents.
        """
        try:
            embeddings = self._embeddings.embed_query(query)
            result = self._get_topk_similar_with_score(embeddings=embeddings, k=k)
            return [doc for doc, _ in result]
        except Exception as error:
            logger.error("Error retrieving similar documents from Snowflake")
            logger.error(error)
            raise error

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError("Method not implemented")

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.similarity_search_with_score(
                    query="What is the first letter in the Greek alphabet?",
                    k=3
                )

        Args:
            query: String query to match against.
            k: Number of matching Documents to return. Defaults to 4.
            kwargs: Vector store specific parameters.

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        try:
            embeddings = self._embeddings.embed_query(query)
            return self._get_topk_similar_with_score(embeddings=embeddings, k=k)
        except Exception as error:
            logger.error("Error retrieving similar documents with score from Snowflake")
            logger.error(error)
            raise error

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Method not implemented")

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.similarity_search_with_relevance_scores(
                    query="What is the first letter in the Greek alphabet?",
                    k=3
                )

        Args:
            query: String query to match against.
            k: Number of matching Documents to return. Defaults to 4.
            kwargs: Vector store specific parameters.

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        score_threshold: Optional[float] = kwargs.get("score_threshold", None)
        try:
            result = self.similarity_search_with_score(query=query, k=k)
            if score_threshold is not None:
                result = [r for r in result if r[1] >= score_threshold]
            return result
        except Exception as error:
            logger.error(
                "Error retrieving similar documents with relevance score from Snowflake"
            )
            logger.error(error)
            raise error

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Method not implemented")

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Example:
            .. code-block:: python

                ids = snowflake_vector_store.similarity_search_by_vector(
                    embedding=[0.3123, 0.6653, ...],
                    k=3
                )

        Args:
            embedding: Embeddings to look up documents similar to.
            k: Number of matching Documents to return. Defaults to 4.
            kwargs: Vector store specific parameters.

        Returns:
            List of Documents most similar to the query vector.
        """
        try:
            result = self._get_topk_similar_with_score(embeddings=embedding, k=k)
            return [doc for doc, _ in result]
        except Exception as error:
            logger.error("Error retrieving similar documents from Snowflake")
            logger.error(error)
            raise error

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError("Method not implemented")

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError("Method not implemented")

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError("Method not implemented")

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError("Method not implemented")

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError("Method not implemented")

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        try:
            return VectorStoreRetriever(vectorstore=self, **kwargs)
        except Exception as error:
            logger.error(
                "Error creating VectorStoreRetriever from SnowflakeVectorStore"
            )
            logger.error(error)
            raise error

    @classmethod
    def from_texts(
        cls: Type[SnowflakeVectorStore],
        connector: SnowflakeConnector,
        embeddings: Embeddings,
        texts: List[str],
        embeddings_dim: int = 768,
        metadatas: Optional[List[dict]] = None,
        table: str = "langchain",
        **kwargs: Any,
    ) -> SnowflakeVectorStore:
        """Return VectorStore initialized from texts and embeddings.

        Example:
            .. code-block:: python

                connector = SnowflakeConnectorBasic(
                    account="account",
                    role="role",
                    user="user",
                    database="database",
                    schema="schema",
                    warehouse="warehouse"
                )

                embeddings = SnowflakeEmbeddings()
                embeddings.connector = connector
                embeddings.model = "e5-base-v2"
                embeddings.embeddings_dim = 768

                snowflake_vector_store = SnowflakeVectorStore.from_texts(
                    connector=connector,
                    embeddings=embeddings,
                    texts=[
                        "Alpha is the first letter of Greek alphabet",
                        "Beta is the second letter of Greek alphabet"
                    ],
                    metadatas=[
                        {"some_key_1": "some value"},
                        {"some_key_2": "some value"}
                    ],
                    embeddings_dim=768,
                    table='embeddings_table'
        Args:
            connector: Snowflake connector instance to use.
            embeddings: Embeddings instance to use.
            texts: List of texts to add to the vectorstore.
            embeddings_dim: Dimensionality of the embeddings. Defaults to 768.
            metadatas: Optional list of metadatas associated with the texts.
            table: Name of the table to use in the Snowflake database.
            kwargs: Vectorstore specific parameters.

        Returns:
            SnowflakeVectorStore instance.
        """
        try:
            vector_store = cls(
                connector=connector,
                embeddings=embeddings,
                embeddings_dim=embeddings_dim,
                table=table,
            )
            vector_store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
            return vector_store
        except Exception as error:
            logger.error("Error creating SnowflakeVectorStore from texts")
            logger.error(error)
            raise error

    @classmethod
    async def afrom_texts(
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError("Method not implemented")

    @classmethod
    async def afrom_documents(
        documents: List[Document], embedding: Embeddings, **kwargs: Any
    ) -> VST:
        raise NotImplementedError("Method not implemented")
