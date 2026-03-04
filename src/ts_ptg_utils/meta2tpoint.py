import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
from astropy.coordinates import ICRS, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
from lsst.summit.utils.butlerUtils import makeDefaultButler
from lsst.daf.butler import DatasetNotFoundError
import lsst_efd_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--day_obs",
    required=True,
    type=int,
    help="Observation day (YYYYMMDD format)",
)
@click.option(
    "--program",
    required=True,
    type=str,
    help="Program name to filter visits (e.g., BLOCK-419)",
)
@click.option(
    "--metadata-dir",
    default="/project/rubintv/LSSTCam/sidecar_metadata",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing metadata JSON files",
)
@click.option(
    "--instrument",
    default="LSSTCam",
    type=str,
    help="Butler instrument name",
)

async def main(
    day_obs: int,
    program: str,
    metadata_dir: str,
    instrument: str,
) -> None:
    """Convert RubinTV metadata to tpoint format.

    Reads metadata from the summit, queries Butler and EFD for visit data,
    and outputs a tpoint-compatible data file.
    """
    location = EarthLocation.from_geodetic(
        lon=-70.747698 * u.deg, lat=-30.244728 * u.deg, height=2663.0 * u.m
    )

    metadata_path = Path(metadata_dir) / f"dayObs_{day_obs}.json"
    with open(metadata_path) as fp:
        metadata = json.load(fp)

    visit_ids = [int(k) for k in metadata if metadata[k]["Program"] == program]
    logger.info(f"Found {len(visit_ids)} visits for program '{program}'")

    efd_client = lsst_efd_client.EfdClient("summit_efd")
    butler = makeDefaultButler(instrument)

    tpoint_data = await process_visits(visit_ids, day_obs, butler, efd_client, location)

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    date_str = f"{day_obs // 10000} {month_names[day_obs // 100 % 100 - 1]} {day_obs % 100:02d}"

    output_path = Path.cwd() / f"simonyi_{day_obs}.dat"
    write_tpoint_file(output_path, date_str, tpoint_data)
    logger.info(f"Wrote {len(tpoint_data)} records to {output_path}")


async def process_visits(
    visit_ids: List[int],
    day_obs: int,
    butler,
    efd_client,
    location,
) -> List[Tuple]:
    """Process visits asynchronously, querying Butler and EFD."""
    tpoint_data = []

    for seq_num in visit_ids:
        data_id = {"day_obs": day_obs, "seq_num": seq_num, "detector": 94}
        try:
            visit_wcs = butler.get("preliminary_visit_image.wcs", data_id)
            if visit_wcs is None:
                logger.warning(f"Skipping {seq_num=}: no preliminary_visit_image.wcs")
                continue

            raw_wcs = butler.get("raw.wcs", data_id)
        except DatasetNotFoundError:
            logger.warning(f"Skipping {seq_num=}: dataset not found")
            continue

        visit_metadata = butler.get("raw.metadata", data_id)
        visit_start_time = Time(
            visit_metadata["MJD-BEG"], format="mjd", scale="tai"
        ).utc
        visit_end_time = Time(visit_metadata["MJD-END"], format="mjd", scale="tai").utc
        observation_time = Time(
            (visit_metadata["MJD-BEG"] + visit_metadata["MJD-END"]) / 2.0,
            format="mjd",
            scale="tai",
        )

        mtmount_elevation = await efd_client.select_time_series(
            "lsst.sal.MTMount.elevation",
            ["actualPosition"],
            start=visit_start_time,
            end=visit_end_time,
        )
        mtmount_azimuth = await efd_client.select_time_series(
            "lsst.sal.MTMount.azimuth",
            ["actualPosition"],
            start=visit_start_time,
            end=visit_end_time,
        )
        mtrotator_position = await efd_client.select_time_series(
            "lsst.sal.MTRotator.rotation",
            ["actualPosition"],
            start=visit_start_time,
            end=visit_end_time,
        )

        altaz_frame = AltAz(
            obstime=observation_time,
            location=location,
            pressure=visit_metadata["PRESSURE"] * u.Pa,
            temperature=visit_metadata["AIRTEMP"] * u.deg_C,
            relative_humidity=visit_metadata["HUMIDITY"],
            obswl=622 * u.nm,
        )

        calexp_sky_center = visit_wcs.pixelToSky(raw_wcs.getPixelOrigin())
        radec_icrs = ICRS(
            calexp_sky_center.getRa().asDegrees() * u.deg,
            calexp_sky_center.getDec().asDegrees() * u.deg,
        )
        radec_icrs_tel = ICRS(
            raw_wcs.getSkyOrigin().getRa().asDegrees() * u.deg,
            raw_wcs.getSkyOrigin().getDec().asDegrees() * u.deg,
        )

        altaz_coord_star = radec_icrs.transform_to(altaz_frame)
        altaz_coord_tel = radec_icrs_tel.transform_to(altaz_frame)

        tpoint_data.append(
            (
                altaz_coord_star.az.deg,
                altaz_coord_star.alt.deg,
                float(mtmount_azimuth.actualPosition.mean()),
                float(mtmount_elevation.actualPosition.mean()),
                float(mtrotator_position.actualPosition.mean()),
            )
        )

    return tpoint_data


def write_tpoint_file(
    output_path: Path,
    date_str: str,
    tpoint_data: List[Tuple],
) -> None:
    """Write tpoint format data file."""
    with open(output_path, "w") as fp:
        fp.write(
            f"""LSST, {date_str}
: ALTAZ
: ROTTEL
-30 14 40.7
"""
        )
        for data in tpoint_data:
            fp.write(
                f"{data[0]:10.6f}\t"
                f"{data[1]:10.6f}\t"
                f"{data[2]:10.6f}\t"
                f"{data[3]:10.6f}\t"
                f"{-data[4]:10.6f}\n"
            )
        fp.write("END")


if __name__ == "__main__":
    asyncio.run(main())
