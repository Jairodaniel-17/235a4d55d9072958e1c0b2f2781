import cx_Oracle
from models.credentials_db_oracle import CredentialsOracle


class OracleConnection:
    def __init__(self, c: CredentialsOracle):
        self.username = c.username
        self.password = c.password
        self.host = c.host
        self.port = c.port
        self.service_name = c.service_name
        self.connection = None

    def connect(self):
        dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.service_name)
        self.connection = cx_Oracle.connect(self.username, self.password, dsn)

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Path: connection/connection_db_mysql.py
