"""Fixtures shared by the test modules."""

from uradata.model import flatten


def project(name="TEST PROJECT", street="TEST ROAD", segment="OCR", transactions=()):
    return {"project": name, "street": street, "marketSegment": segment,
            "transaction": list(transactions)}


def sale(price, sqm, contract="0126", floor="06-10", sale_type="3",
         district="19", property_type="Executive Condominium",
         tenure="99 yrs lease commencing from 2016", units="1"):
    return {"price": str(price), "area": str(sqm), "contractDate": contract,
            "floorRange": floor, "typeOfSale": sale_type, "district": district,
            "propertyType": property_type, "tenure": tenure, "noOfUnits": units}


def build(transactions, **kwargs):
    """Flatten a single synthetic project into Transaction records."""
    return flatten([project(transactions=transactions, **kwargs)])
