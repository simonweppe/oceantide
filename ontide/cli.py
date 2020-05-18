# -*- coding: utf-8 -*-

"""Console script for ontide."""
import sys
import click

import numpy as np
import pandas as pd

from ontide.settings import *
from ontide.otis import predict_tide_point, predict_tide_grid
from ontide.otisoo import OTISoo


@click.group()
def main():
    pass


@main.group()
@click.option("--model", default="tpxo9", help="ID of the gridded constituends model dataset", show_default=True)
@click.option("--catalog", default=ONTAKE_CATALOG, help="Intake catalog that has the source constituents dataset", show_default=True)
@click.option("--namespace", default=ONTAKE_NAMESPACE, help="Intake namespace", show_default=True)
@click.option("--conlist", default=None, show_default=True,
    help="""
List of strings (optional). If supplied, gives a list 
of tidal constituents to include in prediction. If not supplied, 
default from model source will be used.
Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'
""",
)
@click.pass_context
def predict(ctx, model, catalog, namespace, conlist):
    """ Predict tidal elevations and currents.

    """
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["catalog"] = catalog
    ctx.obj["namespace"] = namespace
    ctx.obj["conlist"] = conlist
    click.echo(f"Constituents model ID: {model}")
    click.echo(f"Intake catalog: {catalog}")
    click.echo(f"Instake namespace: {namespace}")

    if conlist != None:
        click.echo(f"Selected constituents: {conlist}")
    else:
        click.echo(f"Selected constituents: ALL available in {model}.zarr")


@predict.command()
@click.option("-x", "--lon", default=None, help="Lon can range from -180 to 180.", required=True)
@click.option("-y", "--lat", default=None, help="Lat can range from -90 to 90.", required=True)
@click.option("-t0", "--start-time", default=None, help="Start time, Ex: 2001-01-01", required=True)
@click.option("-t1", "--end-time", default=None, help="End time, Ex: 2001-02-01", required=True)
@click.option("-o", "--outfile", default=None, help="NetCDF output filename", required=True)
@click.pass_context
def point(ctx, lon, lat, start_time, end_time, outfile):
    """ Predict tides at a point
    """
    time = pd.date_range(start_time, end_time, freq="H")
    predict_tide_point(
        float(lon),
        float(lat),
        time,
        model=ctx.obj["model"],
        catalog=ctx.obj["catalog"],
        namespace=ctx.obj["namespace"],
        outfile=outfile,
        conlist=ctx.obj["conlist"]
    )


@predict.command()
@click.option("-x0", "--x0", default=None, help="Lower left corner lon", show_default=True, required=True)
@click.option("-x1", "--x1", default=None, help="Upper right corner lon", show_default=True, required=True)
@click.option("-y0", "--y0", default=None, help="Lower left corner lat", show_default=True, required=True)
@click.option("-y1", "--y1", default=None, help="Upper right corner lat", show_default=True, required=True)
@click.option("-dx", "--dx", default=None, help="Resolution in X-direction", show_default=True, required=True)
@click.option("-dy", "--dy", default=None, help="Resolution in Y-direction", show_default=True, required=True)
@click.option("-t0", "--start_time", default=None, help="Start time, Ex: 2001-01-01", required=True)
@click.option("-t1", "--end_time", default=None, help="End time, Ex: 2001-02-01", required=True)
@click.option("-o", "--outfile", default=None, help="NetCDF output filename", required=True)
@click.pass_context
def grid(ctx, x0, x1, y0, y1, dx, dy, start_time, end_time, outfile):
    """ Predict tides at a grid
    """
    xi = np.arange(float(x0), float(x1), float(dx))
    yi = np.arange(float(y0), float(y1), float(dy))
    lon, lat = np.meshgrid(xi, yi)
    time = pd.date_range(start_time, end_time, freq="H")
    predict_tide_grid(
        lon,
        lat,
        time,
        model=ctx.obj["model"],
        catalog=ctx.obj["catalog"],
        namespace=ctx.obj["namespace"],
        outfile=outfile,
        conlist=ctx.obj["conlist"]
    )


@main.command()
@click.option("-m", "--model", default=None, help="ID of the gridded constituends model dataset to be created", show_default=True, required=True)
@click.option("-x0", "--x0", default=None, help="Lower left corner lon", show_default=True, required=True)
@click.option("-x1", "--x1", default=None, help="Upper right corner lon", show_default=True, required=True)
@click.option("-y0", "--y0", default=None, help="Lower left corner lat", show_default=True, required=True)
@click.option("-y1", "--y1", default=None, help="Upper right corner lat", show_default=True, required=True)
@click.option("-dx", "--dx", default=None, help="Resolution in X-direction", show_default=True, required=True)
@click.option("-dy", "--dy", default=None, help="Resolution in Y-direction", show_default=True, required=True)
@click.option("-h", "--bathy", default="gebco_2019", show_default=True)
@click.option("-s", "--smooth_fac", default=None, show_default=True)
@click.option("-l", "--hmin", default=2, show_default=True)
@click.option("-b", "--bnd", default="/data/tide/otis_binary/h_tpxo9", show_default=True)
@click.option("-o", "--outfile", default=None, show_default=True, required=True)
def downscale(model, x0, x1, y0, y1, dx, dy, bathy, smooth_fac, hmin, bnd, outfile):
    """ Run OTISoo to produced downscaled constituents grid.
    """
    click.echo(f"Bathy source: {bathy}")
    click.echo(f"Minimum depth: {hmin}")
    click.echo(f"Boundary source: {bnd}")
    click.echo(f"Output file: {outfile}")

    if smooth_fac != None: smooth_fac = float(smooth_fac)
    if hmin != None: hmin = float(hmin)

    otisoo = OTISoo(
        model,
        float(x0),
        float(x1),
        float(y0),
        float(y1),
        dx=float(dx),
        dy=float(dy),
        bnd=bnd,
        bathy=bathy, 
        hmin=hmin,
        smooth_fac=smooth_fac,
        outfile=outfile,
    )

    otisoo.run()



if __name__ == "__main__":
    sys.exit(main(obj={}))  # pragma: no cover
