import logging
from abc import ABC, abstractmethod

try:
    from snowflake.snowpark import Session
except ImportError:
    raise ImportError(
        "`snowflake-snowpark-python` package not found, please install it with "
        "`pip install snowflake-snowpark-python`"
    )

from langchain_core.pydantic_v1 import SecretStr

logger = logging.getLogger(__name__)


class SnowflakeConnector(ABC):
    """
    Abstract base class for Snowflake connector.
    """

    def __init__(
        self,
        account: str,
        role: str,
        user: str,
        database: str,
        schema: str,
        warehouse: str,
    ):
        self._account = account
        self._role = role
        self._user = user
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._session = None
        super().__init__()

    def set_snowflake_logger_params(
        self, level: int, handler: logging.Handler, formatter: logging.Formatter
    ) -> None:
        """Set the logger level for the Snowflake logger.

        Examples:

        .. code-block:: python

            self.set_snowflake_logger_params(
                level=logging.INFO,
                handler=logging.StreamHandler(),
                formatter=logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        Args:
            level (int): The logging level.
            handler: The logging handler.
            formatter: The logging formatter.
        """
        for logger_name in ("snowflake.snowpark", "snowflake.connector"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            handler = handler
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    @abstractmethod
    def connect(self):
        pass


class SnowflakeConnectorBasic(SnowflakeConnector):
    """Wrapper around Snowflake Snowpark for Python.

    This is the default Snowflake connector (using username, password, role etc.).

    To use, you should have the ``snowflake-snowpark-python`` Python package installed.

    Example:
        .. code-block:: python

            from langchain_core.utils import (
                get_from_dict_or_env,
                convert_to_secret_str
            )
            from langchain_community.utilities.snowflake import SnowflakeConnectorBasic

            connector = SnowflakeConnectorBasic(
                account=get_from_dict_or_env(
                    {}, "snowflake_account", "SNOWFLAKE_ACCOUNT"
                ),
                role=get_from_dict_or_env(
                    {}, "snowflake_role", "SNOWFLAKE_ROLE"
                ),
                user=get_from_dict_or_env(
                    {}, "snowflake_user", "SNOWFLAKE_USER"
                ),
                password=convert_to_secret_str(
                    get_from_dict_or_env(
                        {}, "snowflake_password", "SNOWFLAKE_PASSWORD
                    )
                ),
                database=get_from_dict_or_env(
                    {}, "snowflake_database", "SNOWFLAKE_DATABASE"
                ),
                schema=get_from_dict_or_env(
                    {}, "snowflake_schema", "SNOWFLAKE_SCHEMA"
                ),
                warehouse=get_from_dict_or_env(
                    {}, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE"
                )
            )
    """

    def __init__(
        self,
        account: str,
        role: str,
        user: str,
        password: SecretStr,
        database: str,
        schema: str,
        warehouse: str,
    ):
        """Initialize Snowflake connector.

        Args:
            account (str): Snowflake account.
            role (str): Snowflake role.
            user (str): Snowflake user.
            password (SecretStr): Snowflake password.
            database (str): Snowflake database.
            schema (str): Snowflake schema.
            warehouse (str): Snowflake warehouse.
        """
        super().__init__(account, role, user, database, schema, warehouse)
        self._password = password

    def connect(self) -> Session:
        """Connect to Snowflake and get a Session object.

        Returns:
            Session: Snowflake session.
        """
        logger.debug(f"Using {__class__}")
        connection_params = {
            "account": self._account,
            "role": self._role,
            "user": self._user,
            "password": self._password.get_secret_value(),
            "database": self._database,
            "schema": self._schema,
            "warehouse": self._warehouse,
        }
        try:
            self._session = Session.builder.configs(connection_params).create()
            return self._session
        except Exception as error:
            logger.error("Error connecting to Snowflake")
            logger.error(error)
            raise error
