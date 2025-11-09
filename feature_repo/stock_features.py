import os
from datetime import timedelta
from feast import FeatureView, Field, FileSource, Entity
from feast.types import Float32, Int64, String

# Entity definition
stock_entity = Entity(
    name="stock_name",
    join_keys=["stock_name"],
    description="Stock symbol as entity",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "processed_features.parquet"))

# File source
stock_source = FileSource(
    path=DATA_PATH,
    event_timestamp_column="timestamp",
)

# Feature view
stock_features = FeatureView(
    name="stock_features",
    entities=[stock_entity],
    ttl=timedelta(days=1),
    schema=[
        # ✅ include entity explicitly in schema
        Field(name="stock_name", dtype=String),
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=stock_source,
    tags={"team": "mlops", "version": "v1"},
)
