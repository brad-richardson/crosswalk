# Benchmarking Guide

This guide explains how to compare matcher against external tools like Hootenanny.

## GeoParquet to OSM Conversion

Convert any GeoParquet dataset to OSM XML format for use with external conflation tools.

```bash
# Basic conversion (no topology)
python scripts/convert_to_osm.py data/raw/boston_streets.parquet -o boston.osm

# With Overture data and connectors (preserves topology)
python scripts/convert_to_osm.py data/raw/overture_segments.parquet \
    --connectors data/raw/overture_connectors.parquet \
    -o overture.osm

# With custom column names
python scripts/convert_to_osm.py data.parquet \
    --id-column segment_id \
    --class-column road_type \
    --name-column street_name \
    -o output.osm
```

### Conversion Details

The converter:
- Creates OSM `<node>` elements for vertices
- Creates OSM `<way>` elements for each LineString
- Maps the `class` column to `highway=*` tags using standard mappings
- Preserves the `names` column as `name=*` tags
- Adds `matcher:id` tag with the original segment ID for traceability

### Topology Preservation

When the `--connectors` option is provided with Overture connector data, the converter preserves network topology:

- Segments sharing the same `connector_id` will reference the same OSM node
- Connector IDs are hashed to stable negative integers for deterministic output
- Without connectors, each vertex gets a unique node (no topology inference)

### Supported Class Mappings

| Input Class | OSM highway Tag |
|-------------|-----------------|
| motorway, trunk, primary, secondary, tertiary | Same |
| residential, living_street, service | Same |
| footway, sidewalk | footway |
| path, pedestrian, cycleway, track, steps | Same |
| unclassified, (unknown) | unclassified |

## Hootenanny Comparison

[Hootenanny](https://github.com/ngageoint/hootenanny) is a vector conflation tool from NGA that can be used for comparison benchmarking.

> **Note**: Hootenanny installation is complex. There is no pre-built Docker image available.

### Installation Options

#### Option 1: Docker Compose (Recommended)

Use Hootenanny's official docker-compose setup. This builds from source and runs all services in containers:

```bash
# Clone Hootenanny as a sibling to matcher
cd /path/to/matcher/..
git clone https://github.com/ngageoint/hootenanny.git
cd hootenanny

# Start services (first run builds everything - takes 20-40 min)
make -f Makefile.docker up

# Verify it's working
docker compose exec core-services hoot --version
```

Once running, use the Python wrapper:

```python
from pathlib import Path
from matcher.external.hootenanny import conflate

# Files are automatically copied to/from the Hootenanny container
conflate(
    reference=Path("osm/reference.osm"),
    target=Path("osm/target.osm"),
    output=Path("osm/conflated.osm"),
    data_dir=Path("/path/to/matcher/data"),
)
```

Or run commands directly:

```bash
# Copy your OSM files to hootenanny/data/
cp data/osm/*.osm ../hootenanny/data/

# Run conflation
cd ../hootenanny
docker compose exec core-services hoot conflate \
    -D match.creators="HighwayMatchCreator" \
    -D merger.creators="HighwayMergerCreator" \
    /home/hoot/hoot/data/reference.osm \
    /home/hoot/hoot/data/target.osm \
    /home/hoot/hoot/data/conflated.osm
```

To stop services: `make -f Makefile.docker down`

#### Option 2: Vagrant + VirtualBox

Alternative for systems where Docker is problematic:

```bash
sudo apt-get install vagrant virtualbox
git clone https://github.com/ngageoint/hootenanny.git
cd hootenanny
vagrant up
vagrant ssh
# Inside VM: hoot --version
```

#### Option 3: RPM Installation (CentOS 7 only)

For legacy CentOS 7 systems:

```bash
sudo curl -o /etc/yum.repos.d/hootenanny.repo \
    https://s3.amazonaws.com/hoot-repo/el7/release/hoot.repo
sudo yum install hootenanny-core
hoot --version
```

See [hootenanny-rpms](https://github.com/ngageoint/hootenanny-rpms) for details.

### Running Hootenanny Conflation

Once Hootenanny is installed and accessible via the `hoot` command:

```bash
# Convert data to OSM format
python scripts/convert_to_osm.py data/raw/overture_segments.parquet -o reference.osm
python scripts/convert_to_osm.py data/raw/boston_streets.parquet -o target.osm

# Run Hootenanny conflation (roads only)
hoot conflate \
    -D match.creators="HighwayMatchCreator" \
    -D merger.creators="HighwayMergerCreator" \
    reference.osm target.osm conflated.osm
```

### Alternative Tools

If Hootenanny is too complex to install, consider these alternatives for comparison:

- **[RoadMatcher](https://github.com/vividsolutions/roadmatcher)** - Java-based open source tool
- **[JOSM Conflation Plugin](https://josm.openstreetmap.de/)** - Semi-automated conflation in JOSM editor
- **[GraphHopper Map Matching](https://github.com/graphhopper/map-matching)** - For GPS trace to road network matching

## Troubleshooting

### Hootenanny Installation Issues

**Vagrant VM won't start:**
- Ensure VirtualBox is installed and running
- Check that virtualization is enabled in BIOS
- Try `vagrant destroy && vagrant up` for a fresh start

**RPM installation fails on CentOS:**
- Ensure you're running CentOS 7 (not 8+)
- Check that EPEL repository is enabled: `sudo yum install epel-release`

**"hoot: command not found":**
- If using Vagrant, make sure you're inside the VM (`vagrant ssh`)
- If using RPMs, check that `/usr/local/bin` is in your PATH

**Hootenanny conflation hangs:**
- Large datasets may take significant time
- Try with a smaller subset first
- Check system memory (Hootenanny can be memory-intensive)

### Conversion Issues

**"No features parsed from output":**
- Check that input file contains LineString geometries
- Verify CRS is set (EPSG:4326 expected or will be reprojected)

**Empty highway tags:**
- Check that `class` column exists in your data
- Use `--class-column` to specify a different column name

## References

- [Hootenanny documentation](https://github.com/ngageoint/hootenanny): Vector conflation tool
- [MapStitcher paper](https://dl.acm.org/doi/10.1145/2996913.2996999): Graph sampling methodology
- [GraphSamplingToolkit](https://github.com/pfoser/GraphSamplingToolkit): Reference implementation
