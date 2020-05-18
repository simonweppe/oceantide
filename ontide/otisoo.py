# -*- coding: utf-8 -*-
"""OTISoo tools.
  
   Tools to interact with OTIS tidal inverse numerical model
"""

import os, shutil, glob, logging
import numpy as np
from google.cloud import storage, bigquery
from ondata.download.bathy import get_bathy

from ontide.settings import *
from ontide.otis import otisbin2xr


ROOTDIR = os.path.join(os.path.dirname(__file__), "../otisoo")
RUNDIR = "/tmp/otisoo"
DIRTREE = ["exe", "dat", "prm", "repx1", "out", "bathy"]
DBDIR = "/data/tide/otis_binary/DB"
DBBLOB = "otisoo/DB"


class OTISoo(object):
    """ OTISoo inverse tidal model object

        Args:
            dataset_id (str)         ::  name for the regional cons file
            x0, x1, y0, y1 (float)   ::  domain corners (only regular grid supported)
            dx, dy (float)           ::  resolution (preferably dx == dy)
            bathy (str)              ::  intake dataset ID for bathy data
            hmin (int)               ::  minimum depth for the model [m]. MINIMUM ALLOWED VALUE = 2
            smooth_fac (int)         ::  size of the rolling window for bathy smoothing (xr.rolling() is being used across lon and lat)
            bnd (str)                ::  path for the OTIS binary that will serve as a parent model 
                                         (must be the elevation file)
            outfile (str)            ::  path for the output cons zarr or netcdf file (smarts based on file extension). 
                                            IMPORTANT: if "gs" is in the pathname, 
                                            the zarr file will become operational, which means:
                                                - will be uploaded to the operational bucket that contains gridded cons files
                                                - TODO: need to think about how we make intake catalog in sync with the above
                                                - will be registered as a new grid in the cons bounds BQ table 
                                            If left blank, bucket blob name will be assenmbled automatically according to 
                                            default settings
            gcp_sa (str)             ::  GCP service account json file (when interaction with GCP resources is needed)

        Examples
        --------
            - to run an inverse solution, just use the run method:

            otisoo = OTISoo('rag', 174.71, 174.84, -37.84, -37.78, 
                            dx=0.001, dy=0.001, 
                            bnd="/data/tide/otis_binary/hf.NZ2008.out", 
                            outfile="/data/tidecons/rag_cons.nc")
            otisoo.run()

        TODO: bnd should default to None and be automatically detected based 
                on a BQ table with cached corners for each of the downscaled grids
                
    """

    def __init__(
        self,
        dataset_id,
        x0,
        x1,
        y0,
        y1,
        dx=0.01,
        dy=0.01,
        bathy="gebco_2019",
        smooth_fac=None,
        hmin=2,
        bnd="/data/tide/otis_binary/h_tpxo9",
        outfile=None,
        gcp_sa=None,
    ):
        self.dataset_id = dataset_id
        self.localdir = os.path.join(RUNDIR, dataset_id)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.dx, self.dy = dx, dy
        self.bathy = bathy
        self.hmin = hmin
        self.smooth_fac = smooth_fac
        self.bnd = bnd
        self.outfile = outfile or f"gs://oceanum-tide/gridcons/{dataset_id}.zarr"
        self.gcp_sa = gcp_sa

    def make_bathy(self):
        logging.info("Creating bathy and OTISoo grid")

        ds = get_bathy(
            x0=self.x0,
            x1=self.x1,
            y0=self.y0,
            y1=self.y1,
            dx=self.dx,
            dy=self.dy,
            datasource=self.bathy,
            vmin=self.hmin,
            masked=True,
        ).load()

        if self.smooth_fac != None:
            ds = ds.rolling(lon=self.smooth_fac, center=True).mean()
            ds = ds.rolling(lat=self.smooth_fac, center=True).mean()

        # from IPython import embed; embed()

        x, y = np.meshgrid(ds.lon.values, ds.lat.values)
        h = ds.depth.values
        h[np.isnan(h) == 1] = -9  # so the inverse model knows where the mask is

        y, x, h = y.ravel(), x.ravel(), ds.depth.values.ravel()
        dat = np.vstack((y, x, h)).T
        logging.info(dat[:5, :])

        with open(f"{self.localdir}/bathy/bathy.dat", "w", encoding="utf-8") as f:
            for line in range(dat.shape[0]):
                txt = "{:3.6f} {:3.6f} {}\n".format(
                    dat[line, 0], dat[line, 1], int(dat[line, 2])
                )
                f.write(txt)

    def run(self):
        logging.info("Running OTISoo inverse model")
        self._set_environment()
        self.make_bathy()

        for _file in glob.glob(os.path.join(ROOTDIR, "bin/*")):
            os.symlink(
                _file, os.path.join(self.localdir, f"exe/{os.path.basename(_file)}")
            )

        os.chdir(os.path.join(self.localdir, "exe"))
        os.system("./mk_grid -l../bathy/bathy.dat")
        os.system(f"./ob_eval -M{self.bnd}")
        os.system(f"./Fwd_fac")

        logging.info("Writting to {self.outfile}")
        ds = otisbin2xr(
            os.path.join(self.localdir, "prm/grid"),
            os.path.join(self.localdir, "out/h0.df.out"),
            os.path.join(self.localdir, "out/u0.df.out"),
            outfile=self.outfile,
        )

        return ds

    def _set_environment(self):
        self._get_otis_bin()
        logging.info("Setting the environment")
        if os.path.isdir(self.localdir):
            shutil.rmtree(self.localdir)

        os.makedirs(self.localdir)

        for _dir in DIRTREE:
            os.makedirs(os.path.join(self.localdir, _dir))

        shutil.copyfile(
            os.path.join(ROOTDIR, "config/run_param"),
            os.path.join(self.localdir, "exe/run_param"),
        )

        for _file in glob.glob(os.path.join(ROOTDIR, "config/prm/*")):
            shutil.copyfile(
                _file, os.path.join(self.localdir, f"prm/{os.path.basename(_file)}")
            )

        # the below is pretty annoying as the DB files are hardcoded in the fortran code as ../../../DB/{_file}
        if not os.path.isdir("/tmp/DB"):
            shutil.copytree(DBDIR, "/tmp/DB")

    def _auth_storage(self):
        assert (
            self.gcp_sa != None
        ), "gcp_sa argument with GCP service account json file must be provided for this method"
        self.storage_client = storage.Client.from_service_account_json(self.gcp_sa)
        self.bucket = storage.Bucket(self.storage_client, name=BUCKET)

    def _auth_bigquery(self):
        assert (
            self.gcp_sa != None
        ), "gcp_sa argument with GCP service account json file must be provided for this method"
        self.bq_client = bigquery.Client.from_service_account_json(self.gcp_sa)

    def _get_otis_bin(self):
        if not os.path.isdir(DBDIR):
            logging.info("OTISoo binaries not available locally, pulling from bucket")
            self._auth_storage()
            blob = self.bucket.get_blob(DBBLOB)
            blob.download_to_filename(DBDIR)
