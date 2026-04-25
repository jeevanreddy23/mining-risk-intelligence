from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd


GRAVITY_IDENTIFY_URL = 'https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/DMIRS_Imagery_Service/MapServer/8/identify'


def _request_json(url: str, params: dict[str, str]) -> dict:
    with urlopen(f"{url}?{urlencode(params)}") as response:
        return json.loads(response.read().decode('utf-8'))


def sample_gravity(lat: float, lon: float, tolerance: int = 1) -> float | None:
    data = _request_json(
        GRAVITY_IDENTIFY_URL,
        {
            'geometry': f'{lon},{lat}',
            'geometryType': 'esriGeometryPoint',
            'sr': '4326',
            'mapExtent': f'{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}',
            'imageDisplay': '400,400,96',
            'tolerance': str(tolerance),
            'returnGeometry': 'false',
            'f': 'json',
        },
    )
    results = data.get('results', [])
    if not results:
        return None

    attributes = results[0].get('attributes', {})
    for key, value in attributes.items():
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def add_gravity_values(input_csv: Path, output_csv: Path, lat_col: str, lon_col: str) -> Path:
    df = pd.read_csv(input_csv)
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f'Missing latitude/longitude columns: {lat_col}, {lon_col}')

    df['gravity_value'] = [sample_gravity(lat, lon) for lat, lon in zip(df[lat_col], df[lon_col], strict=False)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Sample DMIRS gravity values onto point-based CSV data.')
    parser.add_argument('--input', required=True, help='Input CSV with latitude/longitude columns.')
    parser.add_argument('--output', required=True, help='Output CSV path.')
    parser.add_argument('--lat-col', default='centroid_lat', help='Latitude column name.')
    parser.add_argument('--lon-col', default='centroid_lon', help='Longitude column name.')
    args = parser.parse_args()

    output = add_gravity_values(Path(args.input), Path(args.output), args.lat_col, args.lon_col)
    print(f'Wrote gravity-enriched CSV to {output}')


if __name__ == '__main__':
    main()
