from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime, timezone
from .database import Base


class PredictionAudit(Base):
    """Table to audit all predictions made by the AI model."""
    __tablename__ = "prediction_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Flight input data
    unique_carrier = Column(String, index=True)
    origin = Column(String, index=True)
    dest = Column(String)

    # AI prediction result
    delay_probability = Column(Float)
    is_delayed = Column(Boolean)