#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

import luigi
import psutil
import yaml

from pysisyphus.calculators import ORCA5, XTB
from pysisyphus.drivers.pka import direct_cycle, G_aq_from_h5_hessian
from pysisyphus.helpers import geom_loader, do_final_hessian
from pysisyphus.optimizers.RFOptimizer import RFOptimizer

import pysisyphus.LinearFreeEnergyRelation as LFER

class Params(luigi.Config):
    id_ = luigi.IntParameter()
    name = luigi.Parameter()
    h_ind = luigi.IntParameter()
    is_base = luigi.BoolParameter(default=False)
    charge = luigi.IntParameter(default=0)
    base = luigi.Parameter(default="out")
    acidset = luigi.Parameter()

    @property
    def key(self):
        return f"{self.id_:04d}_{self.name}"

    @property
    def out_dir(self):
        return Path(f"{self.base}_{self.acidset}/{self.key}")

    def get_path(self, fn):
        out_dir = self.out_dir
        is_base_str = "_base" if self.is_base else ""
        fn = f"{self.name}{is_base_str}_{fn}"
        if not out_dir.exists():
            os.mkdir(out_dir)
        return out_dir / fn

    def backup_from_dir(self, dir_, fn, dest_fn=None):
        if dest_fn is None:
            dest_fn = self.get_path(fn)
        shutil.copy(Path(dir_) / fn, dest_fn)

    def calc_charge(self):
        charge = self.charge + (-1 if self.is_base else 0)
        return charge

    @property
    def qm_out_dir(self):
        qm_out_dir = self.get_path("qm_calcs")
        return qm_out_dir


class InputGeometry(Params, luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.get_path("input.xyz"))

    def requires(self):
        if self.is_base:
            return Minimization(self.id_, self.name, self.h_ind, is_base=False, acidset=self.acidset)
        else:
            return None

    def run(self):
        # Derive initial geometry of the base from the optimized acid
        if self.is_base:
            acid_geom = geom_loader(self.input()[0].path)
            # Be sure that it is actually an H atom.
            assert acid_geom.atoms[self.h_ind].lower() == "h"
            geom = acid_geom.get_subgeom_without((self.h_ind,))
        else:
            geom = geom_queue[self.id_]

        with self.output().open("w") as handle:
            handle.write(geom.as_xyz())


class PreMinimization(Params, luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.get_path("preopt.xyz"))

    def requires(self):
        return InputGeometry(self.id_, self.name, self.h_ind, self.is_base, acidset=self.acidset)

    def run(self):
        geom = geom_loader(self.input().path, coord_type="redund")
        geom.set_calculator(
            get_xtb_calc(charge=self.calc_charge(), out_dir=self.qm_out_dir)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            opt = RFOptimizer(
                geom,
                dump=True,
                overachieve_factor=2.0,
                thresh="gau_loose",
                out_dir=tmp_dir,
                max_cycles=500,
            )
            opt.run()
            assert opt.is_converged
            with self.output().open("w") as handle:
                handle.write(geom.as_xyz())


class Minimization(Params, luigi.Task):
    def output(self):
        return (
            luigi.LocalTarget(self.get_path("opt.xyz")),
            luigi.LocalTarget(self.get_path("opt_hessian.h5")),
        )

    def requires(self):
        # Derive the initial base geometry from the optimized acid geometry.
        # Maybe some cycles could be saved when the base is also pre-
        # optimized, but things could also go wrong.
        if self.is_base:
            return InputGeometry(self.id_, self.name, self.h_ind, self.is_base, acidset=self.acidset)
        # Only preoptimize initial acid geometry, not the base.
        else:
            return PreMinimization(self.id_, self.name, self.h_ind, self.is_base, acidset=self.acidset)

    def run(self):
        geom = geom_loader(self.input().path, coord_type="redund")
        geom.set_calculator(
            get_calc(charge=self.calc_charge(), out_dir=self.qm_out_dir)
        )
        final_hess_fn = "final_hessian.h5"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            opt_kwargs_ = {
                "dump": True,
                "overachieve_factor": 4.0,
                "thresh": "gau",
                "out_dir": tmp_dir,
                "max_cycles": 250,
            }
            # Iterate until no imaginary frequencies are present
            for i in range(5):
                opt_kwargs = opt_kwargs_.copy()
                if i > 0:
                    opt_kwargs["hessian_init"] = tmp_dir / final_hess_fn

                opt = RFOptimizer(
                    geom,
                    **opt_kwargs,
                )
                opt.run()
                assert opt.is_converged
                #
                xyz_out, hess_out = self.output()
                with xyz_out.open("w") as handle:
                    handle.write(geom.as_xyz())
                hess_result = do_final_hessian(geom, out_dir=tmp_dir)
                if len(hess_result.neg_eigvals) == 0:
                    self.backup_from_dir(tmp_dir, final_hess_fn, hess_out.path)
                    break
                else:
                    suffix = f".{i:02d}"
                    self.backup_from_dir(tmp_dir, final_hess_fn, hess_out.path + suffix)
                    self.backup_from_dir(
                        tmp_dir, "final_geometry.xyz", xyz_out.path + suffix
                    )
                    # Delete everything in tmp_dir, besides final_hess_fn, as it will be
                    # reused.
                    files = [
                        f
                        for f in tmp_dir.glob("./*")
                        if f.is_file() and f.name != final_hess_fn
                    ]
                    for f in files:
                        os.remove(f)
            else:
                raise Exception("Minimization failed!")


class SolvEnergy(Params, luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.get_path("solv_energy"))

    def requires(self):
        return Minimization(self.id_, self.name, self.h_ind, self.is_base, acidset=self.acidset)

    def run(self):
        geom = geom_loader(self.input()[0].path)
        geom.set_calculator(
            get_solv_calc(charge=self.calc_charge(), out_dir=self.qm_out_dir)
        )
        solv_energy = geom.energy
        with self.output().open("w") as handle:
            handle.write(str(solv_energy))


class DirectCycle(Params, luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.get_path("summary.yaml"))

    def requires(self):
        return (
            Minimization(self.id_, self.name, self.h_ind, is_base=False, acidset=self.acidset),
            Minimization(self.id_, self.name, self.h_ind, is_base=True, acidset=self.acidset),
            SolvEnergy(self.id_, self.name, self.h_ind, is_base=False, acidset=self.acidset),
            SolvEnergy(self.id_, self.name, self.h_ind, is_base=True, acidset=self.acidset),
        )

    def run(self):
        acid_inp, base_inp, acid_solv_fn, base_solv_fn = self.input()
        _, acid_h5 = [inp.path for inp in acid_inp]
        _, base_h5 = [inp.path for inp in base_inp]

        # Load solvated electronic energy from files
        with open(acid_solv_fn.path) as handle:
            acid_solv_en = float(handle.read())
        with open(base_solv_fn.path) as handle:
            base_solv_en = float(handle.read())
        G_aq_H = -6.28 + (-265.9)  # corresponds to the default
        pKa = direct_cycle(acid_h5, base_h5, acid_solv_en, base_solv_en, G_aq_H=G_aq_H)
        print(f"@@@ {self.name}: pka={pKa:.4f}")

        G_acid_aq = G_aq_from_h5_hessian(acid_h5, acid_solv_en)
        G_base_aq = G_aq_from_h5_hessian(base_h5, base_solv_en)
        G_diss_aq = G_base_aq - G_acid_aq

        results = {
            "name": self.name,
            "h_ind": self.h_ind,
            "G_acid_aq": G_acid_aq,
            "G_base_aq": G_base_aq,
            "G_diss_aq": G_diss_aq,
            "pKa_calc": pKa,
        }
        with self.output().open("w") as handle:
            yaml.dump(results, handle)


class DirectCycler(luigi.Task):
    acidlist = luigi.Parameter()
    acidset = luigi.Parameter()
    
    def output(self):
        return luigi.LocalTarget(Path(f"out_{self.acidset}/{self.acidset}_summary.yaml"))
    
    def requires(self):
        acids = yaml.safe_load(self.acidlist)
        
        for id_, (acid, acid_dict) in enumerate(acids.items()):
            h_ind = acid_dict["h_ind"]
            name = acid
            yield DirectCycle(id_=id_, name=name, h_ind=h_ind, acidset=self.acidset)
    
    def run(self):
        res = {}
        for dc in self.input():
            with dc.open() as handle:
                results = yaml.load(handle, Loader=yaml.SafeLoader)
                name = results["name"]
                pKa_calc = results["pKa_calc"]
                # Get the experimental pka values if given (only known for training and validation)
                acidlist = yaml.safe_load(self.acidlist)
                if self.acidset == "targetset": 
                    pka_exp = None
                else:
                    pka_exp = acidlist[name]["pks_exp"]
                res[name] = {
                    "pKa_calc": pKa_calc,
                    "pKa_exp": pka_exp
                    } 
            print("@@@", dc.path, pKa_calc)
        
        summary = {f"{self.acidset}": res}
            
        with self.output().open("w") as handle:
            yaml.dump(summary, handle)

class LFER_Correction(luigi.Task):
    yaml_inp = luigi.Parameter()
    
    def output(self):
        return( luigi.LocalTarget(Path("out_LFER-correction/LFER_summary.yaml")) )
    
    def requires(self):
        with open(self.yaml_inp) as handle:
            run_dict = yaml.load(handle.read(), Loader=yaml.SafeLoader)
        
        # Dump the dictionaries into text, so they can be saved in a luigi parameter.
        trainingset   = yaml.dump(run_dict["trainingset"])
        validationset = yaml.dump(run_dict["validationset"])
        targetset     = yaml.dump(run_dict["targetacids"])
        return( DirectCycler(acidlist=trainingset  , acidset="trainingset"  ),
                DirectCycler(acidlist=validationset, acidset="validationset"),
                DirectCycler(acidlist=targetset    , acidset="targetset"    ) )

    def run(self):
        results = {}
        print("DIRECT CYCLES DONE. RESULT OF COMPUTATION:")
        for dc in self.input():
            with dc.open() as handle:
                res = yaml.load(handle, Loader=yaml.SafeLoader)
                print(yaml.dump(res))
            results = results | res
        
        # DO LFER, validation and correct the target molecules pKa-values
        # 1. Reformat the input data from the previous task
        trainingset = {
            "pKa_calc": [ acid["pKa_calc"] for acid in results["trainingset"].values() ],
            "pKa_exp" : [ acid["pKa_exp"]  for acid in results["trainingset"].values() ],
            "name"    : list( results["trainingset"].keys() ) }
        validationset = {
            "pKa_calc": [ acid["pKa_calc"] for acid in results["validationset"].values() ],
            "pKa_exp" : [ acid["pKa_exp"]  for acid in results["validationset"].values() ],
            "name"    : list( results["validationset"].keys() ) }
        targetset = {
            "pKa_calc": [ acid["pKa_calc"] for acid in results["targetset"].values() ],
            "name"    : list( results["targetset"].keys() ) }
        
        # 2. Do LFER
        # Pass the data to the LFER-model
        # This call also fits the trainingsset with a linear model
        model = LFER.Regression(trainingset, validationset, targetset,
                                xlabel="pKa_calc", ylabel="pKa_exp", namelabel="name")
        
        # 3. Do validation
        mean_square_error, square_errors, validation_prediction = model.validate()
        
        # 4. Apply LFER-correction to target molecules
        pka_corrected = model.lfer_correction()
        
        # 5. Add more information so it can be dumped to file later
        # Add mean_square_error to the pka_corrected dictionary
        summary = {
            "acids": pka_corrected,
            "info": {
                "lfer": {
                    "intercept": float(model().intercept_),
                    "slope": float(model().coef_) 
                    },
                "validation": {
                    "mean_square_error": mean_square_error,
                    "square_errors": square_errors,
                    "acids": validation_prediction
                    }
                }
            }
        
        # 6. Create plot
        # TODO: CREATE A PLOT OF THE LINEAR REGRESSION
        
        print(f"@@@ LFER CORRECTION DONE\n{yaml.dump(summary)}")
        with self.output().open("w") as handle:
            yaml.dump(summary, handle)
            
        

class TaskScheduler(luigi.WrapperTask):
    yaml_inp = luigi.Parameter()
    
    def requires(self):
        return( LFER_Correction(self.yaml_inp) )
    
    def run(self):
        pass
        

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("yaml")
    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    with open(args.yaml) as handle:
        run_dict = yaml.load(handle.read(), Loader=yaml.SafeLoader)
    
    # Get a dict of all acids. Merging all dictionaries with acids.
    acid_list = run_dict["trainingset"] | run_dict["validationset"] | run_dict["targetacids"]
    # Get a list of all geoemetries and abstracted protons
    inputs = list()
    for acid, acid_dict in acid_list.items():
        fn = acid_dict["fn"]
        h_ind = acid_dict["h_ind"]
        assert Path(fn).exists(), f"File '{fn}' does not exist!"
        geom = geom_loader(fn)
        assert (
            geom.atoms[h_ind].lower() == "h"
        ), f"Atom at index {h_ind} in '{fn}' is not a hydrogen atom!"
        print(f"Checked {acid}.")
        inputs.append((fn, h_ind))

    global geom_queue
    geom_queue = [geom_loader(fn) for fn, _ in inputs]

    # Calculator setup
    pal = psutil.cpu_count(logical=False)
    calc_cls = ORCA5
    rdc = run_dict["calc"]
    gas_calc_cls = rdc.pop("type")
    gas_kwargs = rdc
    rdsc = run_dict["solv_calc"]
    solv_calc_cls = rdsc.pop("type")
    solv_kwargs = rdsc
    assert gas_calc_cls == solv_calc_cls
    calc_dict = {
        "orca5": ORCA5,
        "xtb": XTB,
    }
    calc_cls = calc_dict[gas_calc_cls]

    global get_calc

    def get_calc(charge, out_dir):
        return calc_cls(
            pal=pal, charge=charge, out_dir=out_dir, base_name="gas", **gas_kwargs
        )

    global get_solv_calc

    def get_solv_calc(charge, out_dir):
        return calc_cls(
            pal=pal,
            charge=charge,
            out_dir=out_dir,
            base_name="solv",
            **solv_kwargs,
        )

    global get_xtb_calc

    def get_xtb_calc(charge, out_dir):
        return XTB(pal=pal, charge=charge, out_dir=out_dir, base_name="xtb")

    luigi.build(
        (TaskScheduler(args.yaml), ),
        local_scheduler=True,
    )


if __name__ == "__main__":
    run()
