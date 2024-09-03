from typing import Iterator

import pytest
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.snowflake import SnowflakeCortexSQL
from langchain_community.utilities.snowflake import SnowflakeConnectorBasic


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
def snowflake_cortex_sql(
    snowflake_connector_basic: SnowflakeConnectorBasic,
) -> Iterator[SnowflakeCortexSQL]:
    llm = SnowflakeCortexSQL(
        model="mistral-7b", temperature=0.0, top_p=0.0, guardrails=False
    )
    llm.connector = snowflake_connector_basic
    yield llm


# @pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_with_string_prompt_invocation(
    snowflake_cortex_sql: SnowflakeCortexSQL,
) -> None:
    result = snowflake_cortex_sql.invoke(
        "What is the first letter in the Greek Alphabet?"
    )
    assert isinstance(result, str)


# @pytest.mark.requires("snowflake-snowpark-python")
@pytest.mark.enable_socket
def test_with_chat_prompt_template_invocation(
    snowflake_cortex_sql: SnowflakeCortexSQL,
) -> None:
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    chain = prompt | snowflake_cortex_sql
    result = chain.invoke(input="What is the first letter in the Greek Alphabet?")
    assert isinstance(result, str)
