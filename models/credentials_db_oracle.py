# Project created by:  Jairo Daniel Mendoza Torres

from pydantic import BaseModel


class CredentialsOracle(BaseModel):
    """
    Model class for storing database credentials.
    """

    username: str
    host: str
    password: str
    service_name: str
    port: int
