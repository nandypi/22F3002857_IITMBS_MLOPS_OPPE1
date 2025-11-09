from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64
from feast import FileSource

stock_source = FileSource(
    path="../processed_features.csv",
    timestamp_field="timestamp",
)

stock_features = FeatureView(
    name="stock_features",
    entities=[],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=stock_source,
)
