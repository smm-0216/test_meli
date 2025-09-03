from pydantic import BaseModel, Field, StrictInt, StrictFloat, StrictStr


class PredictRequest(BaseModel):
    A: StrictInt = Field(0)
    B: StrictInt = Field(10)
    C: StrictFloat = Field(50257.0)
    D: StrictInt = Field(0)
    E: StrictInt = Field(0)
    F: StrictFloat = Field(0.0)
    G: StrictFloat = Field(0.0)
    H: StrictInt = Field(0)
    I: StrictInt = Field(0)
    J: StrictStr = Field("UY")
    K: StrictFloat = Field(0.8)
    L: StrictInt = Field(0)
    M: StrictInt = Field(3)
    N: StrictInt = Field(1)
    O: StrictInt = Field(0)
    P: StrictInt = Field(5)
    Q: StrictStr = Field("0.00")
    R: StrictStr = Field("0.00")
    S: StrictFloat = Field(7.25)
    Monto: StrictStr = Field("37.51")


class PredictResponse(BaseModel):
    prediction: int
