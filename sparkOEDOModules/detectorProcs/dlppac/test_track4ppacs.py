"""Run with: python -m unittest sparkOEDOModules.detectorProcs.dlppac.test_track4ppacs."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparkOEDOModules.detectorProcs.dlppac import dlppacMain, track4ppacs
from sparkOEDOModules.procModules import constants


class Track4PPACsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]").appName("Track4PPACsTests")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.ansi.enabled", "true")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        with patch.object(dlppacMain, "prm_df", None):
            self.prm_df = dlppacMain.GetGeometryParams(self.spark)
        self.parameters = self.prm_df.orderBy("name").collect()

    def positions(self):
        values = []
        for param in self.parameters:
            z = param.geometry[2]
            values.extend((2.0 + 0.005 * z, -3.0 - 0.002 * z))
        return values

    def position_frame(self, rows):
        columns = [f"{constants.ID_COLNAME} long"]
        columns += [f"{p.name}Get{axis} double" for p in self.parameters for axis in ("X", "Y")]
        return self.spark.createDataFrame(rows, ", ".join(columns))

    def assert_track(self, row):
        for key, expected in (("x0", 2.0), ("y0", -3.0), ("a0", 5.0), ("b0", -2.0)):
            self.assertAlmostEqual(row[key], expected, places=8)

    def test_fit_uses_z_and_allows_independent_missing_axes(self):
        complete = self.positions()
        three_each = complete.copy()
        three_each[0] = three_each[3] = None
        two_x = complete.copy()
        two_x[0] = two_x[2] = None
        two_y = complete.copy()
        two_y[1] = two_y[3] = None
        data = self.position_frame([
            (1, *complete), (2, *three_each), (3, *two_x),
            (4, *two_y), (5, *([None] * 8)),
        ])
        result = track4ppacs.track_4ppacs(data, self.prm_df.orderBy(F.desc("name")))
        rows = result.orderBy(constants.ID_COLNAME).collect()
        self.assertEqual([r[constants.ID_COLNAME] for r in rows], [1, 2])
        for row in rows:
            self.assert_track(row)

    def test_degenerate_z_returns_null_without_dividing_by_zero(self):
        params = self.prm_df.withColumn("geometry", F.array(F.lit(0.0), F.lit(0.0), F.lit(10.0)))
        data = self.position_frame([(1, *self.positions())])
        row = track4ppacs.track_4ppacs(data, params).collect()[0]
        self.assertTrue(all(row[key] is None for key in ("x0", "y0", "a0", "b0")))

    def test_process_adds_tracks_in_full_and_reduced_output(self):
        # Supply calibrated hits; exercise the real position reconstruction and tracking.
        hits = [(2, "unused", 0, 0.0, 1.0)]
        for param in self.parameters:
            z = param.geometry[2]
            signals = {"a": 50.0}
            for axis, position, offset in ((0, 2.0 + 0.005 * z, param.geometry[0]),
                                           (1, -3.0 - 0.002 * z, param.geometry[1])):
                difference = (2.0 * (position + offset) / param.ns2mm[axis]
                              - param.delayoffset[axis] + param.linecalib[axis])
                label = "xy"[axis]
                signals[label + "1"] = (100.0 + difference) / 2.0
                signals[label + "2"] = (100.0 - difference) / 2.0
            hits += [(1, param.name + signal, 0, timing, 1.0) for signal, timing in signals.items()]
        data = self.spark.createDataFrame(
            hits, f"{constants.ID_COLNAME} long, cat string, id long, timing double, charge double"
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(dlppacMain, "prm_df", self.prm_df))
            stack.enter_context(patch.object(dlppacMain, "mapping_df", object()))
            stack.enter_context(patch.object(dlppacMain.tref, "Tref", return_value=None))
            stack.enter_context(patch.object(dlppacMain.mapper, "Map", return_value=data))
            for module, name in ((dlppacMain.tref, "SubtractTref"), (dlppacMain.manipulation, "Validate"),
                                 (dlppacMain.tot, "Tot"), (dlppacMain.calibrator, "ToFloat")):
                stack.enter_context(patch.object(module, name, side_effect=lambda frame, *args: frame))
            for full in (True, False):
                with self.subTest(full=full):
                    result = dlppacMain.Process(data, full, None)
                    rows = result.select(constants.ID_COLNAME, "x0", "y0", "a0", "b0").orderBy(constants.ID_COLNAME).collect()
                    self.assertEqual([r[constants.ID_COLNAME] for r in rows], [1, 2])
                    self.assert_track(rows[0])
                    self.assertTrue(all(rows[1][key] is None for key in ("x0", "y0", "a0", "b0")))


if __name__ == "__main__":
    unittest.main()
