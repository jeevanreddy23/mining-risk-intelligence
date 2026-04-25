from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd


DMIRS_ENDPOINTS = {
    'structures': 'https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Geology_and_Soils_Map/MapServer/1/query',
    'geology': 'https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Geology_and_Soils_Map/MapServer/0/query',
    'gravity': 'https://public-services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/DMIRS_Imagery_Service/MapServer/8/query',
}


def _fetch_json(url: str, params: dict[str, str]) -> dict:
    query = urlencode(params)
    with urlopen(f'{url}?{query}') as response:
        return json.loads(response.read().decode('utf-8'))


def fetch_layer(layer: str, where: str = '1=1', out_fields: str = '*', limit: int = 500) -> pd.DataFrame:
    if layer not in DMIRS_ENDPOINTS:
        raise ValueError(f'Unknown layer: {layer}')

    data = _fetch_json(
        DMIRS_ENDPOINTS[layer],
        {
            'where': where,
            'outFields': out_fields,
            'returnGeometry': 'true',
            'f': 'geojson',
            'resultRecordCount': str(limit),
        },
    )

    features = data.get('features', [])
    rows: list[dict[str, object]] = []
    for feature in features:
        props = feature.get('properties', {}).copy()
        geometry = feature.get('geometry')
        if geometry is not None:
            props['geometry_json'] = json.dumps(geometry)
            props['geometry_type'] = geometry.get('type', '') if isinstance(geometry, dict) else ''
        rows.append(props)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch DMIRS public layer data through ArcGIS REST query endpoints.')
    parser.add_argument('--layer', choices=sorted(DMIRS_ENDPOINTS.keys()), required=True)
    parser.add_argument('--output', required=True, help='Output CSV path.')
    parser.add_argument('--where', default='1=1', help='ArcGIS where clause.')
    parser.add_argument('--fields', default='*', help='Comma-separated output fields.')
    parser.add_argument('--limit', type=int, default=500, help='Maximum number of records to request.')
    args = parser.parse_args()

    df = fetch_layer(layer=args.layer, where=args.where, out_fields=args.fields, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Wrote {len(df)} rows to {output_path}')


if __name__ == '__main__':
    main()
