"""Test Snowflake vectorstore functionality."""

from typing import Iterator, List

import pytest
from langchain_core.documents import Document
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.vectorstores.base import VectorStoreRetriever

from langchain_community.embeddings.snowflake import SnowflakeEmbeddings
from langchain_community.utilities.snowflake import SnowflakeConnectorBasic
from langchain_community.vectorstores.snowflake import SnowflakeVectorStore

SNOWFLAKE_EMBEDDINGS_MODELS = {768: "e5-base-v2", 1024: "nv-embed-qa-4"}
SNOWFLAKE_EMBEDDINGS_TABLE = "LANGCHAIN_UNIT_TEST"
INSERTED_ITEMS_LENGTH = 3

# Make sure you have set the following env variables:
# SNOWFLAKE_ACCOUNT
# SNOWFLAKE_ROLE
# SNOWFLAKE_USER
# SNOWFLAKE_PASSWORD
# SNOWFLAKE_WAREHOUSE
# SNOWFLAKE_DATABASE
# SNOWFLAKE_SCHEMA
# SNOWFLAKE_RSA_KEY_PATH
# SNOWFLAKE_RSA_KEY_PASSWORD


@pytest.fixture(scope="module")
def snowflake_connector_basic() -> Iterator[SnowflakeConnectorBasic]:
    connector = SnowflakeConnectorBasic(
        account=get_from_dict_or_env({}, "snowflake_account", "SNOWFLAKE_ACCOUNT"),
        role=get_from_dict_or_env({}, "snowflake_role", "SNOWFLAKE_ROLE"),
        user=get_from_dict_or_env({}, "snowflake_user", "SNOWFLAKE_USER"),
        password=convert_to_secret_str(
            get_from_dict_or_env({}, "snowflake_password", "SNOWFLAKE_PASSWORD")
        ),
        database=get_from_dict_or_env({}, "snowflake_database", "SNOWFLAKE_DATABASE"),
        schema=get_from_dict_or_env({}, "snowflake_schema", "SNOWFLAKE_SCHEMA"),
        warehouse=get_from_dict_or_env(
            {}, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE"
        ),
    )
    yield connector


@pytest.fixture(scope="module", autouse=True)
def snowflake_embeddings(snowflake_connector_basic: SnowflakeConnectorBasic):
    EMBEDDINGS_DIM = 768
    embeddings = SnowflakeEmbeddings()
    embeddings.connector = snowflake_connector_basic
    embeddings.model = SNOWFLAKE_EMBEDDINGS_MODELS[EMBEDDINGS_DIM]
    embeddings.embeddings_dim = EMBEDDINGS_DIM
    embeddings.show_progress = False
    return embeddings


@pytest.fixture(scope="module", autouse=True)
def snowflake_vector_store(
    snowflake_connector_basic: SnowflakeConnectorBasic,
    snowflake_embeddings: SnowflakeEmbeddings,
):
    EMBEDDINGS_DIM = 768
    vector_store = SnowflakeVectorStore(
        connector=snowflake_connector_basic,
        embeddings=snowflake_embeddings,
        embeddings_dim=EMBEDDINGS_DIM,
        table=SNOWFLAKE_EMBEDDINGS_TABLE,
    )
    return vector_store


@pytest.fixture(scope="module", autouse=True)
def prepare_database(snowflake_connector_basic: SnowflakeConnectorBasic):
    with snowflake_connector_basic.connect() as session:
        session.connection.cursor().execute(f"""
            DROP TABLE IF EXISTS {SNOWFLAKE_EMBEDDINGS_TABLE}
        """)


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_as_retriever(snowflake_vector_store: SnowflakeVectorStore) -> None:
    """Test converting Snowflake vector store to retriever."""

    # Expecting a VectorStoreRetriever instance.
    assert isinstance(
        snowflake_vector_store.as_retriever(search_type="similarity"),
        VectorStoreRetriever,
    )


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_add_texts_insert(snowflake_vector_store: SnowflakeVectorStore) -> None:
    """Test adding text items to Snowflake vector store."""
    # Add text items.
    result = snowflake_vector_store.add_texts(
        texts=[f"Test text {str(i)}" for i in range(1, INSERTED_ITEMS_LENGTH + 1)],
        metadatas=[
            {f"test_key_{str(i)}": f"test_value_{str(i)}"}
            for i in range(1, INSERTED_ITEMS_LENGTH + 1)
        ],
    )

    assert len(result) == INSERTED_ITEMS_LENGTH


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_add_texts_insert_or_update(
    snowflake_vector_store: SnowflakeVectorStore,
) -> None:
    """Test updating 1 and inserting 1 text item to Snowflake vector store."""
    result = snowflake_vector_store.add_texts(
        texts=[
            # Update existing item.
            "Test text 1",
            # Insert new item.
            "Test text 4",
        ],
        metadatas=[
            {"test_key_1": "test_value_1_update"},
            {"test_key_1": "test_value_1"},
        ],
    )

    assert len(result) == 1


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_add_documents_insert(snowflake_vector_store: SnowflakeVectorStore) -> None:
    """Test adding document items to Snowflake vector store."""
    # Add document items.
    documents: List[Document] = []
    for i in range(1, INSERTED_ITEMS_LENGTH + 1):
        document = Document(
            page_content=f"Test document {str(i)}",
            metadata={f"test_key_{str(i)}": f"test_value_{str(i)}"},
        )
        documents.append(document)

    result = snowflake_vector_store.add_documents(documents=documents)

    assert len(result) == INSERTED_ITEMS_LENGTH


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_search_with_similarity_method(
    snowflake_vector_store: SnowflakeVectorStore,
) -> None:
    """Test performing similarity search with the Snowflake vector store."""
    result = snowflake_vector_store.search(
        search_type="similarity", query="Test text 1", k=3
    )

    # Expect 3 results and the first to be 'Test text 1'.
    assert len(result) == 3 and result[0].dict()["page_content"] == "Test text 1"


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_search_with_similarity_score_threshold_method(
    snowflake_vector_store: SnowflakeVectorStore,
) -> None:
    """Test performing similarity search with the Snowflake vector store."""
    result = snowflake_vector_store.search(
        search_type="similarity_score_threshold",
        query="Test text 1",
        score_threshold=1.0,
    )

    # Expect 1 results and the first to be 'Test text 1'.
    assert len(result) == 1 and result[0][0].dict()["page_content"] == "Test text 1"


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_similarity_search(snowflake_vector_store: SnowflakeVectorStore) -> None:
    """Test performing similarity search with the Snowflake vector store."""
    result = snowflake_vector_store.similarity_search(query="Test text 1", top_k=3)

    # Expect that the top similar document is the same as the query document.
    assert result[0].dict()["page_content"] == "Test text 1"


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_similarity_search_with_score(
    snowflake_vector_store: SnowflakeVectorStore,
) -> None:
    """Test performing similarity search with score with the Snowflake vector store."""
    result = snowflake_vector_store.similarity_search_with_score(
        query="Test text 1", top_k=3
    )

    # Expect that the top similar document is the same as the query document
    # and the score to be 1.0.
    assert result[0][0].dict()["page_content"] == "Test text 1" and result[0][1] == 1.0


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_similarity_search_by_vector(
    snowflake_vector_store: SnowflakeVectorStore,
    snowflake_embeddings: SnowflakeEmbeddings,
) -> None:
    """Test performing similarity search by a query vector
    with the Snowflake vector store.
    """
    test_embeddings = snowflake_embeddings.embed_query(input="Test text 1")
    result = snowflake_vector_store.similarity_search_by_vector(
        embedding=test_embeddings, top_k=3
    )

    # Expect that the top similar document is the same as the query document
    # and the score to be 1.0.
    assert result[0].dict()["page_content"] == "Test text 1"


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_similarity_search_with_relevance_scores(
    snowflake_vector_store: SnowflakeVectorStore,
) -> None:
    """Test retrieving similar text items from the Snowflake vector store
    with a relevance score threshold."""
    result = snowflake_vector_store.similarity_search_with_relevance_scores(
        query="Test text 1", top_k=3, relevance_score_threshold=1.0
    )

    # Expect only 1 item to be returned.
    assert result[0][0].dict()["page_content"] == "Test text 1" and result[0][1] == 1.0


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_delete(snowflake_vector_store: SnowflakeVectorStore) -> None:
    """Test deleting text items from the Snowflake vector store."""
    result = snowflake_vector_store.delete(ids=[1, 2, 3])

    # Expect that 3 items are deleted.
    assert True if result else False


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_from_texts(
    snowflake_connector_basic: SnowflakeConnectorBasic,
    snowflake_embeddings: SnowflakeEmbeddings,
) -> None:
    """Test creating a Snowflake vector store from text items."""
    result = SnowflakeVectorStore.from_texts(
        connector=snowflake_connector_basic,
        embeddings=snowflake_embeddings,
        texts=[
            f"Test text from texts {str(i)}"
            for i in range(1, INSERTED_ITEMS_LENGTH + 1)
        ],
        metadatas=[
            {f"test_key_{str(i)}": f"test_value_{str(i)}"}
            for i in range(1, INSERTED_ITEMS_LENGTH + 1)
        ],
        embeddings_dim=768,
        table=SNOWFLAKE_EMBEDDINGS_TABLE,
    )

    assert isinstance(result, SnowflakeVectorStore)
