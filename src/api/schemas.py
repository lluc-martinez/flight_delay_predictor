from pydantic import BaseModel, Field

class FlightRequest(BaseModel):
    UniqueCarrier: str = Field(..., example="WN", description="Airline code (e.g. WN, AA, DL)")
    Origin: str = Field(..., example="LAX", description="Origin airport")
    Dest: str = Field(..., example="JFK", description="Destination airport")
    DayOfWeek: int = Field(..., ge=1, le=7, example=5, description="1=Monday, 7=Sunday")
    Month: int = Field(..., ge=1, le=12, example=8, description="Month of the year")
    Distance: int = Field(..., ge=0, example=2475, description="Distance in miles")
    CRSDepTime: int = Field(..., ge=0, le=2359, example=1830, description="Scheduled departure time HHMM")
    Total_Origin_Congestion: int = Field(..., ge=0, example=45, description="Expected concurrent flights")

class PredictionResponse(BaseModel):
    delay_probability: float = Field(..., example=0.85, description="Delay probability")
    is_delayed: bool = Field(..., example=True, description="Final prediction (True if prob > 0.5)")