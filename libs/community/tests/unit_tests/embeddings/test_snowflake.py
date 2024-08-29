from typing import Iterator

import pytest
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.embeddings.snowflake import SnowflakeEmbeddings
from langchain_community.utilities.snowflake import SnowflakeConnectorBasic

SNOWFLAKE_EMBEDDINGS_MODELS = {768: "e5-base-v2", 1024: "nv-embed-qa-4"}

# Make sure you have set the following env variables:
# SNOWFLAKE_ACCOUNT
# SNOWFLAKE_ROLE
# SNOWFLAKE_USER
# SNOWFLAKE_PASSWORD
# SNOWFLAKE_WAREHOUSE
# SNOWFLAKE_DATABASE
# SNOWFLAKE_SCHEMA


@pytest.fixture(scope="module", autouse=True)
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
def snowflake_embeddings_768(snowflake_connector_basic: SnowflakeConnectorBasic):
    EMBEDDINGS_DIM = 768
    embeddings = SnowflakeEmbeddings()
    embeddings.connector = snowflake_connector_basic
    embeddings.model = SNOWFLAKE_EMBEDDINGS_MODELS[EMBEDDINGS_DIM]
    embeddings.embeddings_dim = EMBEDDINGS_DIM
    embeddings.show_progress = False
    yield embeddings


@pytest.fixture(scope="module", autouse=True)
def snowflake_embeddings_1024(snowflake_connector_basic: SnowflakeConnectorBasic):
    EMBEDDINGS_DIM = 1024
    embeddings = SnowflakeEmbeddings()
    embeddings.connector = snowflake_connector_basic
    embeddings.model = SNOWFLAKE_EMBEDDINGS_MODELS[EMBEDDINGS_DIM]
    embeddings.embeddings_dim = EMBEDDINGS_DIM
    embeddings.show_progress = False
    yield embeddings


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_snowflake_embed_query_768(
    snowflake_embeddings_768: SnowflakeEmbeddings,
) -> None:
    """Test Snowflake embeddings."""
    result = snowflake_embeddings_768.embed_query(input="Test")
    assert result is not None and len(result) == snowflake_embeddings_768.embeddings_dim


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_snowflake_embed_documents_768(
    snowflake_embeddings_768: SnowflakeEmbeddings,
) -> None:
    """Test Snowflake embeddings."""
    result = snowflake_embeddings_768.embed_documents(input=["Test 1", "Test 2"])
    assert (
        result is not None
        and len(result) == 2
        and len(result[0]) == snowflake_embeddings_768.embeddings_dim
    )


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_snowflake_embed_query_1024(
    snowflake_embeddings_1024: SnowflakeEmbeddings,
) -> None:
    """Test Snowflake embeddings."""
    result = snowflake_embeddings_1024.embed_query(input="Test")
    assert (
        result is not None and len(result) == snowflake_embeddings_1024.embeddings_dim
    )


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_snowflake_embed_documents_1024(
    snowflake_embeddings_1024: SnowflakeEmbeddings,
) -> None:
    """Test Snowflake embeddings."""
    result = snowflake_embeddings_1024.embed_documents(input=["Test 1", "Test 2"])
    assert (
        result is not None
        and len(result) == 2
        and len(result[0]) == snowflake_embeddings_1024.embeddings_dim
    )
