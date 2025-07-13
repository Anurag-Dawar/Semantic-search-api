from pydantic import BaseModel

class TextQuery(BaseModel):
    query: str
