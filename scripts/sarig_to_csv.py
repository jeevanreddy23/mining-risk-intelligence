from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


def convert_shapefile_to_csv(input_path: Path, output_path: Path) -> None:
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(epsg=4326)

    centroids = gdf.geometry.centroid
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    gdf["geometry_type"] = gdf.geometry.geom_type
    gdf["wkt_geometry"] = gdf.geometry.to_wkt()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SARIG shapefile or vector layer to ML-ready CSV.")
    parser.add_argument("--input", required=True, help="Path to input shapefile or supported GIS vector file.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    args = parser.parse_args()

    convert_shapefile_to_csv(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
