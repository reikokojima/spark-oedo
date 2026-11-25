"""Fit event-by-event tracks from reconstructed DLPPAC positions."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sparkOEDOModules.procModules import constants


def track_4points(
    detector_df: DataFrame,
    prm_df: DataFrame,
    event_col: str = constants.ID_COLNAME,
) -> DataFrame:
    """Return event ID, x0/y0 (mm) and a0/b0 (mrad) for four PPACs.

    detector_df is the wide position table assembled by dlppacMain.Process() etc.,
    with <name>X and <name>Y columns. prm_df supplies each detector's
    name and geometry=[xz_offset, yz_offset, z_offset]. The offsets are
    Z distances from z_offset to the X and Y measurement planes, respectively.
    X is fitted at z_offset + xz_offset, Y at z_offset + yz_offset.
    Each axis needs at least three non-null positions. The intercept is the
    position at z=0, and the angle is the fitted slope multiplied by 1000.
    """
    parameters = prm_df.select("name", "geometry").collect()
    names = [row["name"] for row in parameters]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("prm_df must contain exactly four distinct detector-plane names")

    required = {event_col}
    required.update(f"{name}{axis}" for name in names for axis in ("X", "Y"))
    missing = required - set(detector_df.columns)
    if missing:
        raise ValueError(f"input detector_df is missing tracking columns: {sorted(missing)}")

    for row in parameters:
        geometry = row["geometry"]
        if geometry is None or len(geometry) != 3 or any(offset is None for offset in geometry):
            raise ValueError(f"Invalid geometry for detector {row['name']!r}: {geometry}")

    def fit_expr(axis):
        axis_index = 0 if axis == "X" else 1
        counts, zs, zzs, values, zvalues = [], [], [], [], []
        for row in parameters:
            value = F.col(f"{row['name']}{axis}")
            valid = value.isNotNull()
            geometry = row["geometry"]
            z = F.lit(float(geometry[2]) + float(geometry[axis_index]))
            counts.append(F.when(valid, 1).otherwise(0))
            zs.append(F.when(valid, z).otherwise(0.0))
            zzs.append(F.when(valid, z * z).otherwise(0.0))
            values.append(F.when(valid, value).otherwise(0.0))
            zvalues.append(F.when(valid, value * z).otherwise(0.0))

        n, sz, szz = sum(counts), sum(zs), sum(zzs)
        sv, szv = sum(values), sum(zvalues)
        denominator = n * szz - sz * sz
        # A fit is undefined when every valid point has the same Z coordinate.
        intercept = F.when(denominator != 0.0, (szz * sv - sz * szv) / denominator)
        slope = F.when(denominator != 0.0, (n * szv - sz * sv) / denominator * 1000.0)
        return intercept, slope, n

    x0, a0, nx = fit_expr("X")
    y0, b0, ny = fit_expr("Y")
    return detector_df.filter((nx >= 3) & (ny >= 3)).select(
        event_col, x0.alias("x0"), y0.alias("y0"),
        a0.alias("a0"), b0.alias("b0"),
    )


def track_2points(
    detector_df: DataFrame,
    prm_df: DataFrame,
    event_col: str = constants.ID_COLNAME,
) -> DataFrame:
    """Extrapolate two X/Y measurements to z=0.

    Input/output formats match track_4points(), using two detector names.
    detector_df contains event_col, <name>X and <name>Y for each detector.
    prm_df contains name and geometry=[x_offset, y_offset, z]. X/Y offsets
    must already be applied to the positions; only geometry[2] is used here.
    Return event_col, x0/y0 (mm) and a0/b0 (slope * 1000, mrad).
    Events need both positions in each axis. Equal Z coordinates yield null
    track values because they do not define an extrapolation.
    """
    parameters = prm_df.select("name", "geometry").collect()
    names = [row["name"] for row in parameters]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("prm_df must contain exactly two distinct detector-plane names")

    required = {event_col}
    required.update(f"{name}{axis}" for name in names for axis in ("X", "Y"))
    missing = required - set(detector_df.columns)
    if missing:
        raise ValueError(f"input detector_df is missing tracking columns: {sorted(missing)}")

    for row in parameters:
        geometry = row["geometry"]
        if geometry is None or len(geometry) != 3 or geometry[2] is None:
            raise ValueError(f"Invalid geometry for detector {row['name']!r}: {geometry}")

    z1, z2 = (float(row["geometry"][2]) for row in parameters)
    dz = F.lit(z2 - z1)

    def extrapolate(axis):
        value1, value2 = (F.col(f"{name}{axis}") for name in names)
        valid = value1.isNotNull() & value2.isNotNull()
        intercept = F.when(dz != 0.0, (value1 * z2 - value2 * z1) / dz)
        slope = F.when(dz != 0.0, (value2 - value1) / dz * 1000.0)
        return intercept, slope, valid

    x0, a0, valid_x = extrapolate("X")
    y0, b0, valid_y = extrapolate("Y")
    return detector_df.filter(valid_x & valid_y).select(
        event_col, x0.alias("x0"), y0.alias("y0"),
        a0.alias("a0"), b0.alias("b0"),
    )
