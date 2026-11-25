"""SRPPAC processor with V1190 segment decoding.

Input: RIDF segment Parquet (event_number, dev, fp, det, mod, binary data),
or the decoded segdata/hits format accepted by srppacMain.py.
Raw RIDF files and block_data Parquet must first be converted to segments.

Run from the repository root with the spark-oedo environment active::

    python -m sparkOEDOModules.detectorProcs.srppac.srppacMain \
        --input /path/to/run_segdata.parquet --full

Configure the Scala decoder JAR in the environment before starting Spark.
"""
from pyspark.sql.types import BinaryType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from sparkOEDOModules.procModules import tref, manipulation, mapper, tot, calibrator, constants, monotoneTableConverter,track
from sparkOEDOModules.detectorProcs.srppac import srppacPosDqdx

#DETECTOR_NAMES = ["sr91","sr92","src1","src2","sr11","sr12"]
DETECTOR_NAMES = ["sr91","sr92","src1","src2","sr11","sr12"]
PLANE_NAMES=["sr9","src","sr1"]

mapping_df = None
tref_mapping_df = None
udf_dict = {}

def LoadCSVFiles(spark: SparkSession):
    """
    Load CSV Files to DataFrames. This function should be called once before calling Process()

    Parameters
    ----------
    spark: SparkSession
    """
    global mapping_df, tref_mapping_df, udf_dict
    if mapping_df is None:
        # 全ファイルの読み込みに成功してから設定を反映する。
        new_mapping = mapper.ReadMapCSV(spark, 'srppac.csv', DETECTOR_NAMES)
        new_tref = tref.ReadCSV(spark)
        new_udfs = {}
        for det in DETECTOR_NAMES:
            new_udfs[det + '_xc0'] = monotoneTableConverter.getConverterUDF(
                spark, 'srppac/' + det + '_xc0.csv')
            new_udfs[det + '_yc0'] = monotoneTableConverter.getConverterUDF(
                spark, 'srppac/' + det + '_yc0.csv')
        mapping_df, tref_mapping_df, udf_dict = new_mapping, new_tref, new_udfs



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
        "_srppac_hit", F.explode(F.expr("decode_v1190_segdata(data)"))
    ).select(
        constants.ID_COLNAME, "dev", "fp", "det",
        F.col("_srppac_hit.geo").alias("geo"),
        F.col("_srppac_hit.channel").alias("ch"),
        F.col("_srppac_hit.measurement").alias("value"),
        F.col("_srppac_hit.edge").alias("edge"),
    )

def GetGeometryParams(spark: SparkSession) -> DataFrame:
    """
    Get geometry and calibration parameters for a given detector

    Parameters
    ----------
    spark: SparkSession

    Returns
    -------
    DataFrame containing position offset(x,y) and geometry(xz,yz,z) parameters.
    Both arrays are converted to double, including integer input values.
    "name" should start with "PLANE_NAMES".
    """
    global prm_df
    detprm = [["sr91",[0.0,0.0],[-4.3,4.3,122.5]], ["sr92",[0.0,0.0],[-4.3,4.3,541.0]],
              ["src1",[50.0,0.0],[-4.3,4.3,-1962.0]], ["src2",[50.0,0.0],[-4.3,4.3,-1462.0]],
              ["sr11",[0.0,0.0],[-4.3,4.3,-560.8]], ["sr12",[0.0,0.0],[-4.3,4.3,-210.8]]]
    detprm = [
        [name, [float(value) for value in offset], [float(value) for value in geometry]]
        for name, offset, geometry in detprm
    ]
    prm_df = spark.createDataFrame(
        detprm, "name string, offset array<double>, geometry array<double>"
    )
    return prm_df


def Process(rawDF: DataFrame, full: bool, require: str) -> DataFrame:
    """
    Main processor function for SRPPAC detectors

    Parameters
    ----------
    rawDF: Input rawdata DataFrame
    full: Flag for full outputs
    require: Only include rows with column "require" is not null

    Returns
    -------
    Output DataFrame
    """

    if mapping_df == None:
        print("Call LoadCSVFiles() before calling Process()")
        return

    # Map tref channels first
    tref_df = tref.Tref(rawDF, tref_mapping_df)

    # Generate timecharge dataframes
    df = mapper.Map(rawDF, mapping_df, [constants.ID_COLNAME,"cat","value","id","edge","dev","fp","det","geo"])
    df = tref.SubtractTref(df,tref_df) # Tref subtraction
    df = manipulation.Validate(df,[-100000,100000])
         # Flip edge for cathode signals (trailingComesFirst=True)
    df = df.withColumn("edge", F.when(F.col("cat").endswith("a"), F.col("edge")).otherwise(1-F.col("edge")))
    df = tot.Tot(df)
    df = calibrator.ToFloat(df,"charge", 0, 0.09765627)
    df = calibrator.ToFloat(df,"timing", 0, 0.09765627)
    srppac_df = rawDF.select(constants.ID_COLNAME).dropDuplicates([constants.ID_COLNAME])

    for ppac in DETECTOR_NAMES:
        offset = prm_df.filter(F.col("name") == ppac).select("offset").first()["offset"]
        df_a = df.filter(F.col("cat") == ppac+"a")
        df_x = df.filter(F.col("cat") == ppac+"x")
        df_y = df.filter(F.col("cat") == ppac+"y")
        pos_x = srppacPosDqdx.srppacPosDqdx(df_x,udf_dict[ppac+"_xc0"],8,46.5,2.55,offset[0],True)
        pos_y = srppacPosDqdx.srppacPosDqdx(df_y,udf_dict[ppac+"_yc0"],8,28.5,2.58,offset[1],True)
        pos_x = pos_x.select(constants.ID_COLNAME,F.col("pos").alias(ppac+"x_pos"))
        pos_y = pos_y.select(constants.ID_COLNAME,F.col("pos").alias(ppac+"y_pos"))
        # Fill event data into an array per event and add the ppac name to the column name
        if full:
            # Full output
            df_a = df_a.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("timing").alias(ppac+"a_timing"),
                F.collect_list("charge").alias(ppac +"a_charge")
            )
            df_x = df_x.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("id").alias(ppac+"x_id"),
                F.collect_list("timing").alias(ppac+"x_timing"),
                F.collect_list("charge").alias(ppac +"x_charge")
            )
            df_x = df_x.join(pos_x, [constants.ID_COLNAME], how="left")
            df_y = df_y.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("id").alias(ppac+"y_id"),
                F.collect_list("timing").alias(ppac+"y_timing"),
                F.collect_list("charge").alias(ppac +"y_charge")
            )
            df_y = df_y.join(pos_y, [constants.ID_COLNAME], how="left")
            # join to the final output data frame
            srppac_df = srppac_df.join(df_a, on=[constants.ID_COLNAME], how="left")
            srppac_df = srppac_df.join(df_x, on=[constants.ID_COLNAME], how="left")
            srppac_df = srppac_df.join(df_y, on=[constants.ID_COLNAME], how="left")
        else:
            # Short output
            df_a = df_a.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("timing").alias(ppac+"a_timing"),
                F.collect_list("charge").alias(ppac +"a_charge")
            )
            pos_x = pos_x.filter(F.col(ppac+"x_pos").isNotNull())
            result_df = pos_x.join(pos_y, [constants.ID_COLNAME], how="left")
            result_df = result_df.join(df_a, [constants.ID_COLNAME], how="left")
            if ppac == require:
                srppac_df = srppac_df.join(result_df, [constants.ID_COLNAME], how="inner")
            else:
                srppac_df = srppac_df.join(result_df, [constants.ID_COLNAME], how="left")

    return srppac_df


def Tracking(srppac_df: DataFrame, prm_df: DataFrame) -> DataFrame:
    """Add track parameters for each prefix in PLANE_NAMES.
    Parameters
    ----------
    srppac_df: Input DataFrame containing <detector>x_pos/y_pos columns
    prm_df : geometry DataFrame for tracking or extrapolation

    Returns
    -------
    Output DataFrame containing reconstructed "x0","y0","a0","b0" at z=0.

    """
    if prm_df is None:
        raise ValueError("Call GetGeometryParams() or pass prm_df before Tracking()")
    
    result_df = srppac_df
    for plane in PLANE_NAMES:
        position_columns = [
            column for column in srppac_df.columns
            if column.startswith(plane) and column.endswith(("x_pos", "y_pos"))
        ]
        if not position_columns:
            continue
        # delete last 5 charactor if column name ends with "x_pos" and "y_pos"
        x_names = {column[:-5] for column in position_columns if column.endswith("x_pos")}
        y_names = {column[:-5] for column in position_columns if column.endswith("y_pos")}
        if x_names != y_names:
            raise ValueError(f"Plane {plane!r} must have matching x_pos/y_pos detector columns")
        if len(x_names) == 4:   # 4points tracking
            track_function = track.track_4points
        elif len(x_names) == 2: # 2point extrapolation
            track_function = track.track_2points
        else:
            raise ValueError(f"Plane {plane!r} requires 2 or 4 x_pos columns; found {len(x_names)}")

        # track.py expects <detector>X/Y rather than <detector>x_pos/y_pos.
        detector_df = srppac_df.select(
            constants.ID_COLNAME,
            *[F.col(column).alias(column[:-5] + column[-5].upper()) for column in position_columns],
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
    parser.add_argument('--output', help='output Parquet path (default: input stem + _srppac.parquet)')
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
    output_path = args.output or input_path[:-len('.parquet')] + '_srppac.parquet'
    if input_path.rstrip('/') == output_path.rstrip('/'):
        parser.error('--output must differ from --input')

    builder = SparkSession.builder.appName('SRPPACDecode')
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
        srppac_df = Process(exploded_df, args.full, args.require)
        result_df=Tracking(srppac_df,prm_df)
        result_df.write.mode('overwrite').parquet(output_path)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
