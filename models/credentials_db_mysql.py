# Project created by:  Jairo Daniel Mendoza Torres

from pydantic import BaseModel


class CredentialsMYSQL(BaseModel):
    """
    Model class for storing database credentials.
    """

    host: str
    user: str
    password: str
    database: str
    port: int

    def cadena_conexion(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
