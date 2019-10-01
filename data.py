from xml.etree import ElementTree


def load_wb_xml(path: str, country_key: str):
    """
    Load xml file in world bank data and extract data for specific country.
    """

    tree = ElementTree.parse(path)

    records = tree.findall(
        ".//record/*[@name='Country or Area'][@key='%s']/.." % country_key)

    data = []
    for record in records:
        year = record.findtext("field[@name='Year']")
        value = record.findtext("field[@name='Value']")
        if year is not None and value:
            data.append((int(year), float(value)))

    return data
