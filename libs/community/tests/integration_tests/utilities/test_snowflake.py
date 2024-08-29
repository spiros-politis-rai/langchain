"""Integration tests for Snowflake."""

from typing import Iterator

import pytest
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.utilities.snowflake import SnowflakeConnectorBasic

# Make sure you have set the following env variables:
# SNOWFLAKE_ACCOUNT
# SNOWFLAKE_ROLE
# SNOWFLAKE_USER
# SNOWFLAKE_PASSWORD
# SNOWFLAKE_WAREHOUSE
# SNOWFLAKE_DATABASE
# SNOWFLAKE_SCHEMA


@pytest.fixture(autouse=True, scope="module")
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
    return connector


@pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_snowflake_connector_basic(
    snowflake_connector_basic: SnowflakeConnectorBasic,
) -> None:
    """Test plain Snowflake connector."""
    with snowflake_connector_basic.connect() as session:
        cursor = session.connection.cursor()
        result = cursor.execute("SELECT 1").fetchall()[0][0]
        assert result == 1
