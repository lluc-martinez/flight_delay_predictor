from pydantic import BaseModel, Field

class FlightRequest(BaseModel):
    UniqueCarrier: str = Field(..., description="Airline code (e.g. WN, AA, DL)", json_schema_extra={"example": "DL"})
    Origin: str = Field(..., description="Origin airport", json_schema_extra={"example": "ATL"})
    Dest: str = Field(..., description="Destination airport", json_schema_extra={"example": "JFK"})
    DayOfWeek: int = Field(..., ge=1, le=7, description="1=Monday, 7=Sunday", json_schema_extra={"example": 5})
    Month: int = Field(..., ge=1, le=12, description="Month of the year", json_schema_extra={"example": 12})
    Distance: int = Field(..., ge=0, description="Distance in miles", json_schema_extra={"example": 760})
    CRSDepTime: int = Field(..., ge=0, le=2359, description="Scheduled departure time HHMM", json_schema_extra={"example": 1830})
    Total_Origin_Congestion: int = Field(..., ge=0, description="Expected concurrent flights", json_schema_extra={"example": 120})

class PredictionResponse(BaseModel):
    delay_probability: float = Field(..., description="Delay probability", json_schema_extra={"example": 0.85})
    is_delayed: bool = Field(..., description="Final prediction (True if prob > 0.5)", json_schema_extra={"example": True})