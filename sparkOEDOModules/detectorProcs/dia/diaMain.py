"""Diamond processor with V1290 segment decoding.

Input: RIDF segment Parquet (event_number, dev, fp, det, mod, binary data),
or decoded segdata/hits Parquet. Convert raw RIDF or block_data first.

Run from the repository root with the spark-oedo environment active::

    python -m sparkOEDOModules.detectorProcs.dia.diaMain \
        --input optics0034_segdata --full

Configure the Scala decoder JAR in the environment before starting Spark.
"""


from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from sparkOEDOModules.procModules import tref, manipulation, mapper, tot, calibrator, constants
from sparkOEDOModules.detectorProcs.dia import twoSidedPlastic

DETECTOR_NAMES = ["dia3"]
#DETECTOR_NAMES = ["dia3","dias2"]

mapping_df = None
tref_mapping_df = None

def LoadCSVFiles(spark: SparkSession):
    """
    Load CSV Files to DataFrames. This function should be called once before calling Process()

    Parameters
    ----------
    spark: SparkSession
    """
    global mapping_df, tref_mapping_df
    if mapping_df is None:
        new_mapping = mapper.ReadMapCSV(spark, "dia1290.csv", DETECTOR_NAMES)
        new_tref = tref.ReadCSV(spark)
        mapping_df, tref_mapping_df = new_mapping, new_tref


def DecodeRawData(spark: SparkSession, rawDF: DataFrame) -> DataFrame:
    """Decode V1290 segments (mod == 25) into flat hits for Process().

    Other module types are excluded before decoding. Already decoded
    segdata/hits input is handled by mapper.ExplodeRawData().
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
        decoder = getattr(getattr(spark._jvm.decoders, "V1290Decoder$"), "MODULE$")
        decoder.registerUDF(spark._jsparkSession)
    except Exception as exc:
        raise RuntimeError(
            "Cannot register decoders.V1290Decoder. Ensure the Scala decoder JAR "
            "is available in Spark's startup configuration."
        ) from exc

    segments = rawDF.filter((F.col("mod") == 25))
    return segments.withColumn(
        "_dia_hit", F.explode(F.expr("decode_v1290_segdata(data)"))
    ).select(
        constants.ID_COLNAME, "dev", "fp", "det",
        F.col("_dia_hit.geo").alias("geo"),
        F.col("_dia_hit.channel").alias("ch"),
        F.col("_dia_hit.measurement").alias("value"),
        F.col("_dia_hit.edge").alias("edge"),
    )


def Process(rawDF: DataFrame, full: bool, require: str) -> DataFrame:
    """
    Main processor function for Diamond detector

    Parameters
    ----------
    rawDF: Input rawdata DataFrame
    full: Flag for full outputs
    require: Only include rows with column "require" is not null

    Returns
    -------
    Output DataFrame
    """
    if mapping_df is None:
        print("Call LoadCSVFiles() before calling Process()")

    # Map tref channels first
    tref_df = tref.Tref(rawDF, tref_mapping_df)

    # Generate timecharge dataframes for each category
    df = mapper.Map(rawDF, mapping_df, [constants.ID_COLNAME, "cat", "value", "id", "edge", "dev", "fp", "det", "geo"])
    df = tref.SubtractTref(df, tref_df)
    df = manipulation.Validate(df,[-200000,200000])
    df = tot.Tot(df)
    df = calibrator.ToFloat(df,"charge", 0, 0.0244140625)
    df = calibrator.ToFloat(df,"timing", 0, 0.0244140625)

    # process for each diamond detector
    detector_df = rawDF.select(constants.ID_COLNAME).dropDuplicates([constants.ID_COLNAME])
    for det in DETECTOR_NAMES:
        df_p = df.filter(F.col("cat") == det+"pad")
        # Aggrigate by events
        df_p = df_p.groupBy(constants.ID_COLNAME).agg(
            F.collect_list("timing").alias(det+"pad_timing"),
            F.collect_list("charge").alias(det +"pad_charge")
        )
        df_l = df.filter(F.col("cat") == det+"stripL")
        df_r = df.filter(F.col("cat") == det+"stripR")
        has_strips = bool(df_l.head(1)) and bool(df_r.head(1))
        if has_strips:
            df_s = twoSidedPlastic.twoSidedPlastic(df_l, df_r, det+"strip", [-100,100])
        
            df_l = df_l.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("id").alias(det+"stripL_id"),
                F.collect_list("timing").alias(det+"stripL_timing"),
                F.collect_list("charge").alias(det +"stripL_charge")
            )
            df_r = df_r.groupBy(constants.ID_COLNAME).agg(
                F.collect_list("id").alias(det+"stripR_id"),
                F.collect_list("timing").alias(det+"stripR_timing"),
                F.collect_list("charge").alias(det +"stripR_charge")
            )

        if full:
            # Full output
            detector_df = detector_df.join(df_p,constants.ID_COLNAME,"left")
            if has_strips:
                detector_df = detector_df.join(df_l,constants.ID_COLNAME,"left")
                detector_df = detector_df.join(df_r,constants.ID_COLNAME,"left")
                detector_df = detector_df.join(df_s,constants.ID_COLNAME,"left")
        else:
            # Reduced output
            if det == require:
                how = "inner"
            else:
                how = "left"
            detector_df = detector_df.join(df_p,constants.ID_COLNAME,how) 
            if has_strips:
                df_s = df_s.select(constants.ID_COLNAME, det+"strip_id", det+"strip_tdiff", det+"strip_tavg", det+"strip_qsqsum")
                detector_df = detector_df.join(df_s,constants.ID_COLNAME,how)

    return detector_df
 
def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='input Parquet path, or a file stem under constants.DATA_PATH')
    parser.add_argument('--output', help='output Parquet path (default: input stem + _dia.parquet)')
    parser.add_argument('--partitions', help='number of partitions', nargs='?', const=1, type=int)
    parser.add_argument('--full', help='output full time charge data', action='store_true')
    parser.add_argument('--require', help='required detector name. Rows without this detector will be deleted in reduced output')
    args = parser.parse_args()
    if args.partitions is not None and args.partitions < 1:
        parser.error('--partitions must be positive')

    if args.input.rstrip('/').endswith('.parquet'):
        input_path = args.input.rstrip('/')
    else:
        input_path = constants.DATA_PATH.rstrip('/') + '/' + args.input + '.parquet'
    output_path = args.output or input_path[:-len('.parquet')] + '_dia.parquet'
    if input_path.rstrip('/') == output_path.rstrip('/'):
        parser.error('--output must differ from --input')

    builder = SparkSession.builder.appName('DIADecode')
    if constants.SERVER_TYPE != 'aws':
        builder = (builder.master(constants.MASTER)
                   .config('spark.driver.memory', '8g')
                   .config('spark.executor.memory', '8g'))
    spark = builder.getOrCreate()
    try:
        raw_df = spark.read.parquet(input_path)
        if args.partitions is not None:
            raw_df = raw_df.repartition(args.partitions)
        exploded_df = DecodeRawData(spark, raw_df)
        LoadCSVFiles(spark)
        detector_df = Process(exploded_df, args.full, args.require)
        detector_df.write.mode('overwrite').parquet(output_path)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
