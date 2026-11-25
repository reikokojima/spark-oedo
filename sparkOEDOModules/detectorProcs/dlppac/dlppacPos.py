"""Position reconstruction from DLPPAC timing differences."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sparkOEDOModules.procModules import constants


def dlppacPos(
    df_x1: DataFrame,
    df_x2: DataFrame,
    df_y1: DataFrame,
    df_y2: DataFrame,
    prm_df: DataFrame,
    ppac: str,
) -> tuple[DataFrame, DataFrame]:
    """Return X and Y data frames for one DLPPAC detector.

    Each input contains the event ID, timing and charge for one signal.
    Process() selects the earliest (rank-1) hit per event and signal first.
    Parameters are selected from prm_df by the detector name in ppac.
    Signal columns are prefixed with ppac; position columns are GetX/GetY.
    An axis requires both of its timing signals. With exchange enabled,
    the Y timing pair supplies GetX and the X timing pair supplies GetY.
    Geometry offsets are applied when reflectx is enabled, as in Process().
    """
    parameters = prm_df.filter(F.col("name") == ppac).select(
        "ns2mm", "delayoffset", "linecalib", "exchange", "reflectx", "geometry"
    ).collect()
    if len(parameters) != 1:
        raise ValueError(
            f"Expected one geometry parameter row for {ppac!r}, got {len(parameters)}"
        )
    param = parameters[0]

    def rename_signal(frame: DataFrame, signal: str) -> DataFrame:
        return frame.select(
            constants.ID_COLNAME,
            F.col("timing").alias(f"{ppac}{signal}_timing"),
            F.col("charge").alias(f"{ppac}{signal}_charge"),
        )

    df_x1 = rename_signal(df_x1, "x1")
    df_x2 = rename_signal(df_x2, "x2")
    df_y1 = rename_signal(df_y1, "y1")
    df_y2 = rename_signal(df_y2, "y2")

    df_x = df_x1.join(df_x2, on=[constants.ID_COLNAME], how="inner")
    df_y = df_y1.join(df_y2, on=[constants.ID_COLNAME], how="inner")

    df_x = df_x.withColumn(
        "dTX",
        F.col(f"{ppac}x1_timing") - F.col(f"{ppac}x2_timing")
        + param.delayoffset[0] - param.linecalib[0],
    ).withColumn("TXsum", F.col(f"{ppac}x1_timing") + F.col(f"{ppac}x2_timing"))
    df_y = df_y.withColumn(
        "dTY",
        F.col(f"{ppac}y1_timing") - F.col(f"{ppac}y2_timing")
        + param.delayoffset[1] - param.linecalib[1],
    ).withColumn("TYsum", F.col(f"{ppac}y1_timing") + F.col(f"{ppac}y2_timing"))

    if param.exchange:
        # Exchange the axis frames so each expression uses its own timing pair.
        df_x, df_y = df_y, df_x
        df_x = df_x.withColumn("GetX", F.col("dTY") * param.ns2mm[1] * 0.5)
        df_y = df_y.withColumn("GetY", F.col("dTX") * param.ns2mm[0] * 0.5)
    else:
        df_x = df_x.withColumn("GetX", F.col("dTX") * param.ns2mm[0] * 0.5)
        df_y = df_y.withColumn("GetY", F.col("dTY") * param.ns2mm[1] * 0.5)

    if param.reflectx:
        df_x = df_x.withColumn("GetX", -F.col("GetX") - param.geometry[0])
    else:
        df_x = df_x.withColumn("GetX", F.col("GetX") - param.geometry[0])

    df_y = df_y.withColumn("GetY", F.col("GetY") - param.geometry[1])
 

    return df_x, df_y
