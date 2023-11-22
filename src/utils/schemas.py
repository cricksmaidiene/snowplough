import pyarrow as pa

all_the_news_raw_schema: pa.Schema = pa.schema(
    [
        pa.field("date", pa.date32()),
        pa.field("year", pa.int64()),
        pa.field("month", pa.int64()),
        pa.field("day", pa.int64()),
        pa.field("author", pa.string()),
        pa.field("title", pa.string()),
        pa.field("article", pa.string()),
        pa.field("url", pa.string()),
        pa.field("section", pa.string()),
        pa.field("publication", pa.string()),
    ]
)