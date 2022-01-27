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

import matplotlib.pyplot as plt
import numpy as np

from pysisyphus.calculators import ORCA5, XTB
from pysisyphus.drivers.pka import direct_cycle, G_aq_from_h5_hessian
from pysisyphus.helpers import geom_loader, do_final_hessian
from pysisyphus.optimizers.RFOptimizer import RFOptimizer

import pysisyphus.LinearFreeEnergyRelation as LFER

class Params(luigi.Config):
    name = luigi.Parameter()
    h_ind = luigi.IntParameter()
    is_base = luigi.BoolParameter(default=False)
    charge = luigi.IntParameter()
    base = luigi.Parameter(default="out")
    acidset = luigi.Parameter()
    fn = luigi.Parameter()

    @property
    def out_dir(self):
        return Path(f"output/{self.base}_{self.name}")

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
            return Minimization(self.name, self.h_ind, is_base=False, acidset=self.acidset, fn=self.fn, charge=self.charge)
        else:
            return None

    def run(self):
        # Derive initial geometry of the base from the optimized acid
        if self.is_base:
            acid_geom = geom_loader(self.input()[0].path)
            # Be sure that it is actually an H atom.
            assert acid_geom.atoms[self.h_ind].lower() == "h", f"Deriving geometry of the base from optimised acid failed! Atom index {self.h_ind} in {self.input()[0].path} is '{acid_geom.atoms[self.h_ind].lower()}' not 'h'."
            geom = acid_geom.get_subgeom_without((self.h_ind,))
        else:
            geom = geom_loader(self.fn)

        with self.output().open("w") as handle:
            handle.write(geom.as_xyz())


class PreMinimization(Params, luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.get_path("preopt.xyz"))

    def requires(self):
        return InputGeometry(self.name, self.h_ind, self.is_base, acidset=self.acidset, fn=self.fn, charge=self.charge)

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
            return InputGeometry(self.name, self.h_ind, self.is_base, acidset=self.acidset, fn=self.fn, charge=self.charge)
        # Only preoptimize initial acid geometry, not the base.
        else:
            return PreMinimization(self.name, self.h_ind, self.is_base, acidset=self.acidset, fn=self.fn, charge=self.charge)

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
                    opt_kwargs.update(
                        {
                            "hessian_init": tmp_dir / final_hess_fn,
                            # Disable overachieve factor to avoid convergence
                            # in the first cycle
                            "overachieve_factor": 0.0,
                        }
                    )

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
        return Minimization(self.name, self.h_ind, self.is_base, acidset=self.acidset, fn=self.fn, charge=self.charge)

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
            Minimization(self.name, self.h_ind, is_base=False, acidset=self.acidset, fn=self.fn, charge=self.charge),
            Minimization(self.name, self.h_ind, is_base=True, acidset=self.acidset, fn=self.fn, charge=self.charge),
            SolvEnergy(self.name, self.h_ind, is_base=False, acidset=self.acidset, fn=self.fn, charge=self.charge),
            SolvEnergy(self.name, self.h_ind, is_base=True, acidset=self.acidset, fn=self.fn, charge=self.charge),
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
            "pKa_calc": pKa
        }
        with self.output().open("w") as handle:
            yaml.dump(results, handle)


class DirectCycler(luigi.Task):
    acidlist = luigi.Parameter()
    acidset = luigi.Parameter()
    
    def output(self):
        return luigi.LocalTarget(Path(f"output/{self.acidset}_summary.yaml"))
    
    def requires(self):
        acids = yaml.safe_load(self.acidlist)
        
        for acid, acid_dict in acids.items():
            h_ind = acid_dict["h_ind"]
            name = acid
            fn = acid_dict["fn"]
            try:
                charge = acid_dict["charge"]
            except KeyError:
                charge = 0
            yield DirectCycle(name=name, h_ind=h_ind, acidset=self.acidset, fn=fn, charge=charge)
    
    def run(self):
        # Get section with the acids from the input file
        acids = yaml.safe_load(self.acidlist)
        
        # Organise the result of the DirectCylce Task
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
                    "pKa_exp": pka_exp,
                    "group": acids[name]["group"]
                    } 
            print("@@@", dc.path, pKa_calc)
        
        summary = {f"{self.acidset}": res}
            
        with self.output().open("w") as handle:
            yaml.dump(summary, handle)

class LFER_Correction(luigi.Task):
    yaml_inp = luigi.Parameter()
    
    # List with names of the plot, that this task will generate.
    LIST_OF_IMAGE_FILES = ["trainingsPlot", "LFER"]
    
    # Helper function to get optional parameters (=settings) from the yaml file.
    # Define funcion to get the settings either from a nested dictionary.
    # The settings can be defined on the first layer of settings and these settings will be used for all plots.
    # If there is a subdictionary with the name of the plot, the function returns the setting specific for that plot and not the global settings.
    # If nothing is defined, return the default value.
    def getPyplotSettingsHandler(self, plotname):
        # Read the input file
        with open(self.yaml_inp) as handle:
            inputDict = yaml.load(handle.read(), Loader=yaml.SafeLoader)
        # Get the section about pyplot
        settings = inputDict.get("pyplot", {})
        # Define a function, that grabs the correct value.
        def getSettings(key, default):
            try:
                # Try to access the section specific to the current plot
                return settings[plotname].get(key, default)
            except KeyError:
                # If this section does not exist, grab the setting from the top level
                return settings.get(key, default)
        return getSettings
    
    def output(self):
        
        # Define the summary file as target
        targets = {"summaryFile": luigi.LocalTarget(Path("output/LFER_summary.yaml")) }
        
        # APPEND IMAGE FILES TO THE TARGET LIST
        for name in self.LIST_OF_IMAGE_FILES:    
            imageSettings = self.getPyplotSettingsHandler(name)
            # Get the file format for the output file of the plot (image file extension) default to .png
            imageOutputSuffix = imageSettings("format", ".png")
            # Add a leading . to the file extension if not existing.
            if not imageOutputSuffix.startswith("."): "." + imageOutputSuffix
            # Add the image file with the correct suffix and name to the targets.
            targets.update({name: luigi.LocalTarget(Path(f"output/plot_{name}{imageOutputSuffix}"), format=luigi.format.Nop) })
        
        # Return the target list
        return targets
        
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
            "group"   : [ acid["group"]    for acid in results["trainingset"].values() ],
            "name"    : list( results["trainingset"].keys() ) }
        validationset = {
            "pKa_calc": [ acid["pKa_calc"] for acid in results["validationset"].values() ],
            "pKa_exp" : [ acid["pKa_exp"]  for acid in results["validationset"].values() ],
            "group"   : [ acid["group"]    for acid in results["validationset"].values() ],
            "name"    : list( results["validationset"].keys() ) }
        targetset = {
            "pKa_calc": [ acid["pKa_calc"] for acid in results["targetset"].values() ],
            "group"   : [ acid["group"]    for acid in results["targetset"].values() ],
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
        
        print(f"@@@ LFER CORRECTION DONE\n{yaml.dump(summary)}")
        with self.output()["summaryFile"].open("w") as handle:
            yaml.dump(summary, handle)
        
        # 6. Create plot
        print("PLOT LFER ...")
        
        # Create image file
        # Save image to temporary file. Saving it later to the actual output, because pyplot I can't find a way to combine pyplot.savefig with luigi.LocalTarget
        # Create the temporary file with the same suffix as the final output file.
        def saveFig(outputFile, figure):
            # Get a temporary file to save the image initially to.
            tmpFile = tempfile.NamedTemporaryFile(suffix=Path(outputFile.path).suffix)
            # Save the image to the output file
            figure.savefig(tmpFile.name)
            # Read the image from file into a variable.
            with open(tmpFile.name, "rb") as file:
                image = file.read()
            # Write the variable to the actual output file.
            with outputFile.open("w") as handle:
                handle.write(image)
        
        #
        #   CREATE FIRST PLOT: TRAININGSET PLOT
        #
        # Get settings for the first plot             
        getSettings = self.getPyplotSettingsHandler(self.LIST_OF_IMAGE_FILES[0])
        
        # Create an empty plot.
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')
        
        # Add title and labels
        ax1.set_title(getSettings("title", "Trainingsset"))
        ax1.set_xlabel(getSettings("xlabel", r'experimental $\mathrm{p}K_a$'))
        ax1.set_ylabel(getSettings("ylabel", r'calculated $\mathrm{p}K_a$'))
        
        # Add a grid
        ax1.grid()
        
        # Add diagonal line
        ax1.plot(trainingset["pKa_exp"], trainingset["pKa_exp"], c='r', label=r"$f(x)=x$")
        
        # Plot training set
        groups = list(set(trainingset["group"]))
        for group in groups:
            x = [ x for index, x in enumerate(trainingset["pKa_exp"]) if trainingset["group"][index] == group ]
            y = [ y for index, y in enumerate(trainingset["pKa_calc"]) if trainingset["group"][index] == group ]
            ax1.scatter(x=x, y=y, s=10, label=group)
        
        # Add the legend
        fig.legend(loc=getSettings("legendLoc", "upper right"))
        
        # Save the figure to the output file.
        saveFig(self.output()["trainingsPlot"], fig)
        print("TRAININGS PLOT DONE")
        
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
