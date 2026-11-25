"""Fit event-by-event tracks from reconstructed DLPPAC positions."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sparkOEDOModules.procModules import constants


def track_4ppacs(
    dlppac_df: DataFrame,
    prm_df: DataFrame,
    event_col: str = constants.ID_COLNAME,
) -> DataFrame:
    """Return event ID, x0/y0 (mm) and a0/b0 (mrad) for four PPACs.

    dlppac_df is the wide position table assembled by dlppacMain.Process(),
    with <name>GetX and <name>GetY columns. prm_df supplies each detector's
    name and geometry=[x_offset, y_offset, z]. Position reconstruction has
    already applied the X/Y offsets; both axes are fitted against geometry[2].
    Each axis needs at least three non-null positions. The intercept is the
    position at z=0, and the angle is the fitted slope multiplied by 1000.
    """
    parameters = prm_df.select("name", "geometry").collect()
    names = [row["name"] for row in parameters]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("prm_df must contain exactly four distinct PPAC names")

    required = {event_col}
    required.update(f"{name}Get{axis}" for name in names for axis in ("X", "Y"))
    missing = required - set(dlppac_df.columns)
    if missing:
        raise ValueError(f"dlppac_df is missing tracking columns: {sorted(missing)}")

    for row in parameters:
        geometry = row["geometry"]
        if geometry is None or len(geometry) != 3 or geometry[2] is None:
            raise ValueError(f"Invalid geometry for PPAC {row['name']!r}: {geometry}")

    def fit_expr(axis):
        counts, zs, zzs, values, zvalues = [], [], [], [], []
        for row in parameters:
            value = F.col(f"{row['name']}Get{axis}")
            valid = value.isNotNull()
            z = F.lit(float(row["geometry"][2]))
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
    return dlppac_df.filter((nx >= 3) & (ny >= 3)).select(
        event_col, x0.alias("x0"), y0.alias("y0"),
        a0.alias("a0"), b0.alias("b0"),
    )
