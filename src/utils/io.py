"""A module for defining input / output configurations"""

from typing import Literal
import pandas as pd
import contextlib
from pathlib import Path
import numpy as np
import os
import re
import shutil

import fsspec

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pyarrow_fs

from deltalake import DeltaTable, write_deltalake


class FileSystemHandler:
    """A class to handle saving and loading files from a filesystem."""

    def __init__(
        self,
        fs_type: Literal["local", "s3"] = "local",
        local_dir_path: str | None = None,
        s3_bucket: str | None = None,
    ):
        """Initialize the FileSystemHandler with a specific filesystem type.

        Provides a unified API for using a local or cloud storage filesystem to store tables
        and files for this project

        Args:
            fs_type (Literal["local", "s3"], optional): The type of filesystem to use. Defaults to "local".
            local_dir_path (str | None): If fs_type is local, then which local directory to save delta tables to
            s3_bucket (str): Name of the S3 bucket if fs_typeis set to s3
        """
        self.fs_type = fs_type
        """The type of filesystem to use."""
        self.arrow_fs = pyarrow_fs
        """The pyarrow filesystem to use."""
        self.fs, self.parquet_fs = self.get_filesystem()

        if fs_type == "s3":
            self.initialize_s3(s3_bucket=s3_bucket)
            os.environ["AWS_S3_ALLOW_UNSAFE_RENAME"] = "True"

        elif fs_type == "local":
            self.initialize_local(local_dir=local_dir_path)

        self.CATALOG: str = f"{self.BASE_PATH}/catalog"
        """Catalog URL for Delta Schemas"""

    def get_filesystem(self):
        """Return a filesystem instance based on `fs_type`."""
        if self.fs_type == "local":
            return fsspec.filesystem("file"), self.arrow_fs.LocalFileSystem()
        if self.fs_type == "s3":
            return fsspec.filesystem("s3"), self.arrow_fs.S3FileSystem()

        raise Exception(f"Unsupported filesystem type: {self.fs_type}")

    def initialize_local(self, local_dir: str | None = None):
        """Initialize Local Configs for Delta I/O."""
        local_data_folder: str = (
            Path(__file__).parent.parent.parent / "data"
            if local_dir is None
            else local_dir
        )
        os.makedirs(local_data_folder, exist_ok=True)
        self.BASE_PATH = local_data_folder
        """The Base Local Path for reading and writing delta tables"""

    def initialize_s3(self, s3_bucket: str | None = None):
        """Initialize AWS S3 Cloud Storage Configs for Delta I/O."""
        self.S3_BUCKET: str = "snowplough-mids" if s3_bucket is None else s3_bucket
        """The S3 BUCKET to use by default"""
        self.BASE_PATH: str = f"s3://{self.S3_BUCKET}"
        """The Base S3 Path for reading and writing delta tables"""

    def read_delta(
        self, table: str, catalog_name: str, as_pandas: bool = False
    ) -> DeltaTable | pd.DataFrame:
        """Read a delta table object or equivalent pandas dataframe."""
        if not table.endswith(".delta"):
            table = table + ".delta"
        table_path: str = self.CATALOG +  f"/{catalog_name}/{table}"

        dt = DeltaTable(table_path)
        return dt.to_pandas() if as_pandas else dt

    def write_delta(
        self,
        dataframe: pd.DataFrame,
        table: str,
        catalog_name: str,
        mode: Literal["append", "overwrite"] = "append",
        partition_by: list[str] | None = None,
    ) -> bool:
        """Write a pandas dataframe as a delta table."""
        if not table.endswith(".delta"):
            table = table + ".delta"
        table_path: str = self.CATALOG +  f"/{catalog_name}/{table}"

        write_deltalake(table_path, dataframe, mode=mode, partition_by=partition_by)
        return True
    
    def clear_delta(self, table: str, catalog_name: str, purge: bool = True) -> dict:
        """Removes a delta table and its schema from the catalog location."""
        if not table.endswith(".delta"):
            table = table + ".delta"
        table_path: str = self.CATALOG +  f"/{catalog_name}/{table}"
        table_del: dict = DeltaTable(table_path).delete()

        if purge:
            for sub_part in self.fs.ls(table_path):
                self.fs.rm(sub_part, recursive=True)

        return table_del
    

    def listdir(
        self,
        location: str,
        pattern: str | None = None,
        files_only: bool = False,
        recursive: bool = False,
    ) -> list:
        """List all paths in a directory at a specific location.

        Args:
            location (str): The location of the directory to list paths from.
            pattern (Optional[str], optional): A regex pattern to filter paths by. Defaults to None.
            files_only (bool, optional): Whether to only list paths that are files. Defaults to False.
            recursive (bool, optional): Whether to list all sub-paths recursively. Defaults to False.

        Returns:
            list: The list of paths.
        """
        paths = []

        if self.fs_type == "s3" and location.startswith("s3://"):
            location = location[5:]

        file_selector = self.arrow_fs.FileSelector(location, recursive=recursive)
        paths = self.parquet_fs.get_file_info(file_selector)
        if files_only:
            paths = list(filter(lambda file: file.is_file, paths))

        if pattern:
            paths = list(
                filter(lambda path: re.search(re.compile(pattern), path.path), paths),
            )

        return paths


def is_databricks_env() -> bool:
    """Verifies if the current computing environment is on databricks"""
    with contextlib.suppress(NameError):
        if "dbutils" in globals():
            return True

    return False


def normalize_parquet_dataframe(parquet_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize a parquet dataframe. This function will convert all array-like iterables (tuple, set, np.array) to lists.

    Args:
        parquet_dataframe (pd.DataFrame): The dataframe read from a parquet file.
           Input from `get_records_from_s3_parquet_object`.

    Returns:
        pd.DataFrame: The normalized dataframe.
    """
    df = parquet_dataframe.copy()

    for col in df.columns:
        candidate_values = df[col].dropna()

        if not candidate_values.empty and isinstance(
            candidate_values.iloc[0],
            (np.ndarray, list, tuple, set),
        ):
            df[col] = (
                df[col]
                .dropna()
                .apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x),
                )
            )

    return df


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize null values in a dataframe.

    For ex. if a column has a value of `None`, `np.nan`, `""`, `[]`, or `{}`,
    then it will be replaced with `np.nan`.

    Args:
        df (pd.DataFrame): The dataframe to normalize

    Returns:
        pd.DataFrame: The null-normalized dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Incorrect argument types specified for function")

    df_local: pd.DataFrame = df.copy()
    df_local = df_local.applymap(
        lambda cell: np.nan
        if (type(cell) is list or type(cell) is dict) and len(cell) == 0
        else cell,
    )
    df_local = df_local.replace("", np.nan)
    df_local = df_local.replace([np.nan], [None])
    return df_local