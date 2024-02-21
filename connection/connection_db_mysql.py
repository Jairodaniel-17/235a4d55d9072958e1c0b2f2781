# Project created by:  Jairo Daniel Mendoza Torres

import mysql.connector

from models.CredentialsDataBase import CredentialsDataBase


class MySQLConnection:
    def __init__(self, credentials: CredentialsDataBase):
        self.host = credentials.host
        self.user = credentials.user
        self.password = credentials.password
        self.database = credentials.database
        self.port = credentials.port
        self.connection = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
        )
        # Retorna la instancia de MySQLConnection
        return self

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, type, value, traceback):
        self.disconnect()

    def execute_query(self, query: str) -> list:
        cursor = self.connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def table_exists(self, table_name: str) -> bool:
        query = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        result = self.execute_query(query)
        return result[0][0] == 1
