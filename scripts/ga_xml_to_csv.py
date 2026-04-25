from __future__ import annotations

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


NS = {"sc": "http://geofon.gfz.de/ns/seiscomp-schema/0.14"}


def _find_text(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path, NS)
    return found.text.strip() if found is not None and found.text else ""


def parse_event(xml_path: Path) -> dict[str, str]:
    root = ET.parse(xml_path).getroot()
    event = root.find(".//sc:event", NS)
    origin = root.find(".//sc:origin", NS)
    magnitude = root.find(".//sc:magnitude", NS)
    description = root.find(".//sc:event/sc:description", NS)

    event_id = ""
    if event is not None:
        event_id = event.attrib.get("publicID", "")

    data = {
        "event_id": event_id,
        "event_type": _find_text(event, "sc:type"),
        "region_name": _find_text(description, "sc:text"),
        "origin_time_utc": _find_text(origin, "sc:time/sc:value"),
        "latitude": _find_text(origin, "sc:latitude/sc:value"),
        "longitude": _find_text(origin, "sc:longitude/sc:value"),
        "depth_km": "",
        "magnitude_type": _find_text(magnitude, "sc:type"),
        "magnitude_value": _find_text(magnitude, "sc:magnitude/sc:value"),
        "magnitude_uncertainty": _find_text(magnitude, "sc:magnitude/sc:uncertainty"),
        "evaluation_mode": _find_text(origin, "sc:evaluationMode"),
        "evaluation_status": _find_text(origin, "sc:evaluationStatus"),
        "station_count": _find_text(origin, "sc:quality/sc:associatedStationCount"),
        "phase_count": _find_text(origin, "sc:quality/sc:associatedPhaseCount"),
        "standard_error": _find_text(origin, "sc:quality/sc:standardError"),
        "azimuthal_gap": _find_text(origin, "sc:quality/sc:azimuthalGap"),
        "min_horizontal_uncertainty_km": _find_text(origin, "sc:uncertainty/sc:minHorizontalUncertainty"),
        "max_horizontal_uncertainty_km": _find_text(origin, "sc:uncertainty/sc:maxHorizontalUncertainty"),
        "creation_agency": _find_text(origin, "sc:creationInfo/sc:agencyID"),
    }

    depth_m = _find_text(origin, "sc:depth/sc:value")
    if depth_m:
        data["depth_km"] = f"{float(depth_m):.6f}"

    return data


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GA SeisComP XML event files into a flat CSV.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more XML files.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    rows = [parse_event(Path(path)) for path in args.input]
    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
