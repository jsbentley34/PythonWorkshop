from typing import List, Tuple
from xml.etree import ElementTree


def load_wb_xml(path: str, region_key: str) -> List[Tuple[int, float]]:
    """
    Parse time series stored in world bank format

    :param path: File containing data
    :param region_key: Queried region code. World bank does use 3-letter
                       codes for each country/region
    :return: List of paired (year, value associated with given year)
    """

    tree = ElementTree.parse(path)

    records = tree.findall(
        ".//record/*[@name='Country or Area'][@key='%s']/.." % region_key
    )

    data = []
    for record in records:
        year = record.findtext("field[@name='Year']")
        value = record.findtext("field[@name='Value']")
        if year is not None and value:
            data.append((int(year), float(value)))

    return data
