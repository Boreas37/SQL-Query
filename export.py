"""
MSSQL -> Parquet streaming exporter (pymssql / FreeTDS).

Reads the query from SQL_FILE, streams rows in batches and writes each batch
to a Parquet file via pyarrow.ParquetWriter so the whole result set never
has to live in memory at once.
"""

import logging
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymssql


def env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        print(f"missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return val  # type: ignore[return-value]


def parse_server(raw: str) -> tuple[str, int]:
    # Kabul edilen biçimler: "host", "host,1433", "host:1433"
    raw = raw.strip()
    if "," in raw:
        host, _, port = raw.partition(",")
        return host.strip(), int(port.strip() or "1433")
    if ":" in raw:
        host, _, port = raw.partition(":")
        return host.strip(), int(port.strip() or "1433")
    return raw, 1433


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("export")

    sql_file = env("SQL_FILE", "sorgu.sql")
    output_file = env("OUTPUT_FILE", "output.parquet")
    batch_size = int(env("BATCH_SIZE", "20000"))
    sleep_between = float(env("SLEEP_BETWEEN_BATCHES", "0"))
    compression = env("PARQUET_COMPRESSION", "snappy")
    row_group_rows = int(env("PARQUET_ROW_GROUP_ROWS", "50000"))
    login_timeout = int(env("LOGIN_TIMEOUT", "30"))
    query_timeout = int(env("QUERY_TIMEOUT", "0"))  # 0 = sınırsız

    server_raw = env("MSSQL_SERVER", required=True)
    database = env("MSSQL_DATABASE", required=True)
    user = env("MSSQL_USER", required=True)
    password = env("MSSQL_PASSWORD", required=True)
    tds_version = env("TDS_VERSION", "7.3")  # eski sunucular için 7.1 dene

    host, port = parse_server(server_raw)

    with open(sql_file, "r", encoding="utf-8-sig") as f:
        query = f.read().strip()
    if not query:
        log.error("query file %s is empty", sql_file)
        return 2

    log.info("connecting to MSSQL %s:%d (TDS %s)...", host, port, tds_version)
    conn = pymssql.connect(
        server=host,
        port=port,
        user=user,
        password=password,
        database=database,
        login_timeout=login_timeout,
        timeout=query_timeout,
        tds_version=tds_version,
        autocommit=True,
    )

    cursor = conn.cursor()
    # OLTP'yi rahatsız etmeyelim.
    cursor.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
    cursor.execute("SET NOCOUNT ON")

    log.info("executing query from %s (batch=%d)", sql_file, batch_size)
    t_exec = time.time()
    cursor.execute(query)
    log.info("query started in %.1fs; streaming rows", time.time() - t_exec)

    columns = [c[0] for c in cursor.description]

    writer: pq.ParquetWriter | None = None
    target_schema: pa.Schema | None = None
    buffered: list[pa.Table] = []
    buffered_rows = 0
    total = 0
    t_start = time.time()

    def flush() -> None:
        nonlocal buffered, buffered_rows
        if not buffered:
            return
        table = pa.concat_tables(buffered)
        writer.write_table(table)  # type: ignore[union-attr]
        buffered = []
        buffered_rows = 0

    try:
        while True:
            t_fetch = time.time()
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            df = pd.DataFrame.from_records(list(rows), columns=columns)
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                target_schema = table.schema
                writer = pq.ParquetWriter(output_file, target_schema, compression=compression)
            else:
                table = table.cast(target_schema, safe=False)  # type: ignore[arg-type]

            buffered.append(table)
            buffered_rows += table.num_rows
            if buffered_rows >= row_group_rows:
                flush()

            total += len(rows)
            log.info(
                "fetched %d rows in %.2fs (total=%d, elapsed=%.1fs)",
                len(rows),
                time.time() - t_fetch,
                total,
                time.time() - t_start,
            )

            if sleep_between > 0:
                time.sleep(sleep_between)

        flush()
    finally:
        if writer is not None:
            writer.close()
        cursor.close()
        conn.close()

    log.info("done. %d rows written to %s in %.1fs", total, output_file, time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
