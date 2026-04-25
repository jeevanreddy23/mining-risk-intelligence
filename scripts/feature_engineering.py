from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError('Input layer has no CRS defined.')
    return gdf.to_crs(epsg=3857)


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f'Layer is empty: {path}')
    return gdf


def _geometry_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    metric = _to_metric(gdf)
    out = gdf.copy()
    out['geometry_type'] = out.geometry.geom_type
    out['centroid_lon'] = out.to_crs(epsg=4326).geometry.centroid.x
    out['centroid_lat'] = out.to_crs(epsg=4326).geometry.centroid.y
    out['area_m2'] = metric.geometry.area
    out['perimeter_m'] = metric.geometry.length
    out['wkt_geometry'] = out.geometry.to_wkt()
    return out


def _distance_to_nearest(source: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, column_name: str) -> pd.Series:
    source_metric = _to_metric(source)
    targets_metric = _to_metric(targets)
    union = targets_metric.geometry.union_all()
    return source_metric.geometry.distance(union).rename(column_name)


def _point_density(source: gpd.GeoDataFrame, points: gpd.GeoDataFrame, radius_m: float, column_name: str) -> pd.Series:
    source_metric = _to_metric(source)
    points_metric = _to_metric(points)
    spatial_index = points_metric.sindex
    counts: list[int] = []
    for geom in source_metric.geometry:
        search_area = geom.buffer(radius_m)
        candidate_ids = list(spatial_index.intersection(search_area.bounds))
        if not candidate_ids:
            counts.append(0)
            continue
        candidates = points_metric.iloc[candidate_ids]
        counts.append(int(candidates.geometry.intersects(search_area).sum()))
    return pd.Series(counts, index=source.index, name=column_name)


def build_features(
    geology_path: Path,
    minerals_path: Path,
    drillholes_path: Path,
    output_path: Path,
    structures_path: Path | None = None,
) -> Path:
    geology = _read_layer(geology_path)
    minerals = _read_layer(minerals_path)
    drillholes = _read_layer(drillholes_path)
    structures = _read_layer(structures_path) if structures_path else None

    features = _geometry_features(geology)
    features['distance_to_mineral_m'] = _distance_to_nearest(features, minerals, 'distance_to_mineral_m')
    features['drillhole_count_1km'] = _point_density(features, drillholes, 1000.0, 'drillhole_count_1km')
    features['drillhole_count_5km'] = _point_density(features, drillholes, 5000.0, 'drillhole_count_5km')

    if structures is not None:
        features['distance_to_structure_m'] = _distance_to_nearest(features, structures, 'distance_to_structure_m')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.drop(columns='geometry').to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build ML-ready geology feature table from GeoVIEW or SARIG layers.')
    parser.add_argument('--geology', required=True, help='Path to interpreted geology or lithology layer.')
    parser.add_argument('--minerals', required=True, help='Path to MINEDEX or mineral occurrences layer.')
    parser.add_argument('--drillholes', required=True, help='Path to drillholes or collar layer.')
    parser.add_argument('--output', required=True, help='Path to output CSV.')
    parser.add_argument('--structures', help='Optional path to structures, faults, or lineaments layer.')
    args = parser.parse_args()

    build_features(
        geology_path=Path(args.geology),
        minerals_path=Path(args.minerals),
        drillholes_path=Path(args.drillholes),
        output_path=Path(args.output),
        structures_path=Path(args.structures) if args.structures else None,
    )


if __name__ == '__main__':
    main()
