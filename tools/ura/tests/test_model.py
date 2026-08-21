import math
import unittest

from uradata.model import (SQM_TO_SQFT, flatten, floor_midpoint, lease_commencement_year,
                           month_index, parse_contract_date, year_month)

from .helpers import build, project, sale


class TestContractDate(unittest.TestCase):
    def test_parses_mmyy(self):
        self.assertEqual(parse_contract_date("0126"), (2026, 1))
        self.assertEqual(parse_contract_date("1225"), (2025, 12))

    def test_rejects_malformed(self):
        for bad in (None, "", "126", "012026", "abcd", "1326", "0026"):
            self.assertIsNone(parse_contract_date(bad), bad)

    def test_month_index_is_ordered_and_subtractable(self):
        self.assertEqual(month_index("0226") - month_index("0126"), 1)
        self.assertEqual(month_index("0126") - month_index("1225"), 1)
        self.assertEqual(month_index("0126") - month_index("0125"), 12)

    def test_year_month_display(self):
        self.assertEqual(year_month("0126"), "2026-01")
        self.assertEqual(year_month(None), "?")


class TestTenureAndFloor(unittest.TestCase):
    def test_lease_year(self):
        self.assertEqual(
            lease_commencement_year("99 yrs lease commencing from 2016"), 2016)
        self.assertIsNone(lease_commencement_year("Freehold"))
        self.assertIsNone(lease_commencement_year(None))

    def test_floor_midpoint(self):
        self.assertEqual(floor_midpoint("11-15"), 13)
        self.assertEqual(floor_midpoint("01-05"), 3)
        # Landed transactions carry '-' and must not become floor zero.
        self.assertIsNone(floor_midpoint("-"))
        self.assertIsNone(floor_midpoint(None))


class TestFlatten(unittest.TestCase):
    def test_derives_sqft_and_psf(self):
        rows = build([sale(1_988_000, 108.05)])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row.sqft, 108.05 * SQM_TO_SQFT, places=6)
        self.assertAlmostEqual(row.psf, row.price / row.sqft, places=9)

    def test_drops_unparseable_rows_rather_than_guessing(self):
        raw = [project(transactions=[
            sale(1_000_000, 100),
            {"price": "not a number", "area": "100"},
            {"price": "1000000"},                      # missing area
            sale(0, 100),                              # non-positive price
            sale(1_000_000, 0),                        # non-positive area
        ])]
        self.assertEqual(len(flatten(raw)), 1)

    def test_carries_project_level_fields_onto_every_row(self):
        rows = build([sale(1_000_000, 100), sale(1_100_000, 100)],
                     name="RIVERCOVE RESIDENCES", segment="OCR")
        self.assertTrue(all(r.project == "RIVERCOVE RESIDENCES" for r in rows))
        self.assertTrue(all(r.segment == "OCR" for r in rows))

    def test_handles_empty_and_missing_transaction_lists(self):
        self.assertEqual(flatten([]), [])
        self.assertEqual(flatten([{"project": "X"}]), [])
        self.assertEqual(flatten([{"project": "X", "transaction": None}]), [])

    def test_sale_type_naming(self):
        self.assertEqual(build([sale(1, 1, sale_type="1")])[0].sale_type_name, "New Sale")
        self.assertEqual(build([sale(1, 1, sale_type="3")])[0].sale_type_name, "Resale")
        self.assertTrue(build([sale(1, 1, sale_type="3")])[0].is_resale())
        self.assertFalse(build([sale(1, 1, sale_type="1")])[0].is_resale())

    def test_unit_count_falls_back_to_one(self):
        self.assertEqual(build([sale(1, 1, units=None)])[0].units, 1)
        self.assertEqual(build([sale(1, 1, units="4")])[0].units, 4)
