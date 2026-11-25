"""DLPPAC processor with V1190 segment decoding.

Input: RIDF segment Parquet (event_number, dev, fp, det, mod, binary data),
or the decoded segdata/hits format accepted by dlppacMain.py.
Raw RIDF files and block_data Parquet must first be converted to segments.

Run from the repository root with the spark-oedo environment active::

    python -m sparkOEDOModules.detectorProcs.dlppac.dlppacMain \
        --input /path/to/run_segdata(.parquet) --full

Configure the Scala decoder JAR in the environment before starting Spark.
"""


from pyspark.sql.types import BinaryType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sparkOEDOModules.procModules import tref, manipulation, mapper, tot, calibrator, constants, track
from sparkOEDOModules.detectorProcs.dlppac import dlppacPos

# ReadMapCSV matches category prefixes, including all signals of each PPAC.
DETECTOR_NAMES = ["f31A", "f31B", "f32A", "f32B","fe71","fe81"]
PLANE_NAMES=["f3"]

mapping_df = None
tref_mapping_df = None
prm_df = None

def LoadCSVFiles(spark: SparkSession):
    """
    Load CSV Files to DataFrames. This function should be called once before calling Process()

    Parameters
    ----------
    spark: SparkSession
    """
    global mapping_df, tref_mapping_df
    if mapping_df is None:
        # 全ファイルの読み込みに成功してから設定を反映する。
        new_mapping = mapper.ReadMapCSV(spark, 'dlppac.csv', DETECTOR_NAMES)
        new_tref = tref.ReadCSV(spark)
        mapping_df, tref_mapping_df = new_mapping, new_tref



def DecodeRawData(spark: SparkSession, rawDF: DataFrame) -> DataFrame:
    """Return flat hits in the format used by Process().

    V1190 segments (mod == 24) are decoded with the existing Scala UDF.
    Other module types are excluded before decoding. The segment metadata
    is retained; channel/measurement become ch/value for mapper and tref.
    Already decoded segdata is handled by the original ExplodeRawData().
    """
    if constants.ID_COLNAME not in rawDF.columns:
        raise ValueError(f"Input is missing event ID column: {constants.ID_COLNAME}")
    if "segdata" in rawDF.columns:
        return mapper.ExplodeRawData(rawDF)

    required = {"dev", "fp", "det", "mod", "data"}
    missing = required - set(rawDF.columns)
    if missing:
        raise ValueError(
            f"Input is missing segment columns: {sorted(missing)}. "
            "Provide RIDF segment Parquet or decoded segdata/hits Parquet."
        )
    if not isinstance(rawDF.schema["data"].dataType, BinaryType):
        raise ValueError("The segment data column must have BinaryType.")

    try:
        decoder = getattr(getattr(spark._jvm.decoders, "V1190Decoder$"), "MODULE$")
        decoder.registerUDF(spark._jsparkSession)
    except Exception as exc:
        raise RuntimeError(
            "Cannot register decoders.V1190Decoder. Ensure the Scala decoder JAR "
            "is available in Spark's startup configuration."
        ) from exc

    segments = rawDF.filter(F.col("mod") == 24)
    return segments.withColumn(
        "_dlppac_hit", F.explode(F.expr("decode_v1190_segdata(data)"))
    ).select(
        constants.ID_COLNAME, "dev", "fp", "det",
        F.col("_dlppac_hit.geo").alias("geo"),
        F.col("_dlppac_hit.channel").alias("ch"),
        F.col("_dlppac_hit.measurement").alias("value"),
        F.col("_dlppac_hit.edge").alias("edge"),
    )

def GetGeometryParams(spark: SparkSession) -> DataFrame:
    """
    Get geometry and calibration parameters for a given detector

    Parameters
    ----------
    spark: SparkSession

    Returns
    -------
    DataFrame containing geometry and calibration parameters.
    Numeric arrays are converted to double, including integer input values.
    "name" should start with "PLANE_NAMES".
    """
    global prm_df
    columns=["name", "ns2mm","delayoffset", "linecalib","exchange","reflectx","geometry","TXSumLimit","TYSumLimit"]

    detprm = [["f31A",[1.242,-1.231],[0.46,-1.06],[-4.91,-3.63],  0, 0,[ -4.3,4.3,-14.4],[-9999., 9999.],[-9999., 9999.]],
        ["f31B",[1.259,-1.246],[1.07,-0.51],[-11.51,16.34],  0, 0,[ 4.3,-4.3,14.4],   [-9999., 9999.],[-9999., 9999.]],
        ["f32A",[1.241,-1.237],[1.37,-0.21],[6.3435,9.3776], 0, 0,[ -4.3,4.3,875.6],[-9999., 9999.],[-9999., 9999.]],
        ["f32B",[1.236,-1.232],[1.2,-0.27], [5.0054,7.6443], 0, 0,[ 4.3,-4.3,904.4],[-9999., 9999.],[-9999., 9999.]],
        ["fe71",[1.212,1.228],[-1.24775,-0.985438],[0.0,0.0], 0, 0,[ 0.0,0.0,182.8],  [-9999., 9999.],[-9999., 9999.]],
        ["fe72",[1.257,1.257],[0.05,0.04],[4.02,0.661], 0, 0,[ 0.0,0.0,502.8],  [-9999., 9999.],[-9999., 9999.]],
        ["fe81", [1.226,1.221],[0.75,0.63], [0.21,2.17], 0, 0,[ 0.0,0.0,-880.85],[-9999., 9999.],[-9999., 9999.]],
    ]
    detprm = [
        [[float(value) for value in values] if isinstance(values, (list, tuple)) else values
         for values in row]
        for row in detprm
    ]
    prm_df = spark.createDataFrame(detprm, columns)
    return prm_df

def Process(rawDF: DataFrame, full: bool, require: str) -> DataFrame:
    """
    Main processor function for DLPPAC detectors

    Parameters
    ----------
    rawDF: Input rawdata DataFrame
    full: Flag for full outputs
    require: Only include rows with column "require" is not null

    Returns
    -------
    Output DataFrame containing reconstructed GetX/GetY positions.
    Pass this DataFrame to Tracking() to add track parameters.
    """

    if mapping_df == None:
        print("Call LoadCSVFiles() before calling Process()")
        return

    if prm_df == None:
        print("Call GetGeometryParams() before calling Process()")
        return
    

    # Map tref channels first
    tref_df = tref.Tref(rawDF, tref_mapping_df)

    # Generate timecharge dataframes
    df = mapper.Map(rawDF, mapping_df, [constants.ID_COLNAME,"cat","value","id","edge","dev","fp","det","geo"])
    df = tref.SubtractTref(df,tref_df)                 # Tref subtraction
    df = manipulation.Validate(df,[-100000,100000])    # Timing validation
    df = tot.Tot(df)
    # Keep the earliest hit per event and signal before joining detector data.
    # Rank before ToFloat adds random jitter to the measured timing.
    hit_window = Window.partitionBy(constants.ID_COLNAME, "cat").orderBy(
        F.col("timing").asc_nulls_last(), F.col("charge").desc_nulls_last(),
        F.col("id").asc_nulls_last(),
    )
    df = df.withColumn("rank", F.row_number().over(hit_window))
    df = df.filter(F.col("rank") == 1).drop("rank")
    df = calibrator.ToFloat(df,"charge", 0, 0.09765627)
    df = calibrator.ToFloat(df,"timing", 0, 0.09765627)

    dlppac_df = rawDF.select(constants.ID_COLNAME).dropDuplicates([constants.ID_COLNAME])


    for ppac in DETECTOR_NAMES:
        df_a = df.filter(F.col("cat") == ppac+"a")
        df_x1 = df.filter(F.col("cat") == ppac+"x1")
        df_x2 = df.filter(F.col("cat") == ppac+"x2")
        df_y1 = df.filter(F.col("cat") == ppac+"y1")
        df_y2 = df.filter(F.col("cat") == ppac+"y2")

        df_a = df_a.select(
            constants.ID_COLNAME,
            F.col("timing").alias(ppac+"a_timing"),
            F.col("charge").alias(ppac+"a_charge"),
        )
        pos_x, pos_y = dlppacPos.dlppacPos(
            df_x1, df_x2, df_y1, df_y2, prm_df, ppac
        )
        # Keep reconstructed columns unique across detectors, including exchanged axes.
        for column in ("GetX", "GetY", "dTX", "dTY", "TXsum", "TYsum"):
            pos_x = pos_x.withColumnRenamed(column, ppac+column)
            pos_y = pos_y.withColumnRenamed(column, ppac+column)

        if full:
            # Full output
            dlppac_df = dlppac_df.join(df_a, on=[constants.ID_COLNAME], how="left")
            dlppac_df = dlppac_df.join(pos_x, on=[constants.ID_COLNAME], how="left")
            dlppac_df = dlppac_df.join(pos_y, on=[constants.ID_COLNAME], how="left")
        else:
            # Short output
            pos_x = pos_x.filter(F.col(ppac+"GetX").isNotNull())
            result_df = pos_x.join(pos_y, on=[constants.ID_COLNAME], how="left")
            result_df = result_df.select(
                        constants.ID_COLNAME,
                        *[f"{ppac}{axis}" for axis in ("GetX", "GetY")],
                    )
            if ppac == require:
                dlppac_df = dlppac_df.join(result_df, on=[constants.ID_COLNAME], how="inner")
            else:
                dlppac_df = dlppac_df.join(result_df, on=[constants.ID_COLNAME], how="left")
    return dlppac_df

def Tracking(dlppac_df: DataFrame, prm_df: DataFrame) -> DataFrame:
    """Add track parameters for each prefix in PLANE_NAMES.
    Parameters
    ----------
    dlppac_df: Input DataFrame containing "GetX","GetY" columns
    prm_df : geometry DataFrame for tracking or extrapolation

    Returns
    -------
    Output DataFrame containing reconstructed "x0","y0","a0","b0" at z=0.

    """
    if prm_df is None:
        raise ValueError("Call GetGeometryParams() or pass prm_df before Tracking()")
    
    result_df = dlppac_df
    for plane in PLANE_NAMES:
        position_columns = [
            column for column in dlppac_df.columns
            if column.startswith(plane) and column.endswith(("GetX", "GetY"))
        ]
        if not position_columns:
            continue
        # delete last 4 charactors if column name ends with "GetX" and "GetY"
        x_names = {column[:-4] for column in position_columns if column.endswith("GetX")}
        y_names = {column[:-4] for column in position_columns if column.endswith("GetY")}
        if x_names != y_names:
            raise ValueError(f"Plane {plane!r} must have matching GetX/GetY detector columns")
        if len(x_names) == 4: # 4points tracking
            track_function = track.track_4points
        elif len(x_names) == 2: # 2point extrapolation
            track_function = track.track_2points
        else:
            raise ValueError(f"Plane {plane!r} requires 2 or 4 GetX columns; found {len(x_names)}")

        # track.py expects <detector>X/Y rather than <detector>GetX/GetY.
        detector_df = dlppac_df.select(
            constants.ID_COLNAME,
            *[F.col(column).alias(column[:-4] + column[-1]) for column in position_columns],
        )
        plane_geom_df = prm_df.filter(F.col("name").isin(sorted(x_names)))
        track_df = track_function(detector_df, plane_geom_df).select(
            constants.ID_COLNAME,
            *[F.col(column).alias(plane + column) for column in ("x0", "y0", "a0", "b0")],
        )
        result_df = result_df.join(track_df, on=[constants.ID_COLNAME], how="left")

    return result_df

 
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='input Parquet path, or a file stem under constants.DATA_PATH')
    parser.add_argument('--output', help='output Parquet path (default: input stem + _dlppac.parquet)')
    parser.add_argument('--partitions', help='number of partitions', nargs='?', const=1, type=int)
    parser.add_argument('--full', help='output full time charge data', action='store_true')
    parser.add_argument('--require', help='required ppac name. Rows without this ppac column will be deleted')
    args = parser.parse_args()
    if args.partitions is not None and args.partitions < 1:
        parser.error('--partitions must be positive')

    if args.input.rstrip('/').endswith('.parquet'):
        input_path = args.input.rstrip('/')
    else:
        input_path = constants.DATA_PATH.rstrip('/') + '/' + args.input + '.parquet'
    output_path = args.output or input_path[:-len('.parquet')] + '_dlppac.parquet'
    if input_path.rstrip('/') == output_path.rstrip('/'):
        parser.error('--output must differ from --input')

    builder = SparkSession.builder.appName('DLPPACDecode')
    if constants.SERVER_TYPE != 'aws':
        builder = (builder.master(constants.MASTER)
                   .config('spark.driver.memory', '20g')
                   .config('spark.executor.memory', '20g'))
    spark = builder.getOrCreate()
    try:
        raw_df = spark.read.parquet(input_path)
        if args.partitions is not None:
            raw_df = raw_df.repartition(args.partitions)
        exploded_df = DecodeRawData(spark, raw_df)
        LoadCSVFiles(spark)
        GetGeometryParams(spark)
        dlppac_df = Process(exploded_df, args.full, args.require)
        result_df=Tracking(dlppac_df,prm_df)
        result_df.write.mode('overwrite').parquet(output_path)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
