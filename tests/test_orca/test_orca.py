from concurrent.futures import ThreadPoolExecutor
from os import kill
from signal import SIGTERM
from subprocess import check_output
from time import sleep

import pytest

from pysisyphus.helpers import geom_loader
from pysisyphus.init_logging import init_logging
from pysisyphus.calculators import ORCA
from pysisyphus.testing import using


@pytest.fixture
def retry_geom():
    geom = geom_loader("lib:h2o.xyz")
    calc_kwargs = {
        "keywords": "hf sto-3g",
        "retry_calc": 1,
    }
    calc = ORCA(**calc_kwargs)
    geom.set_calculator(calc)
    return geom


@using("orca")
def test_orca_check_termination(retry_geom, this_dir):
    calc = retry_geom.calculator
    normal_term_out = this_dir / "normal_term.orca.out"

    assert calc.check_termination(normal_term_out)

    with open(normal_term_out) as handle:
        text = handle.read()
    assert calc.check_termination(text)

    fail_term_out = this_dir / "fail_term.orca.out"
    assert not calc.check_termination(fail_term_out)


@using("orca")
def test_print_capabilities(retry_geom):
    retry_geom.calculator.print_capabilities()


@using("orca")
def test_orca_restart():
    """Spawn two threads. One executes ORCA with retry_calc=1, the other thread
    kills the ORCA process. Then retry_calc kicks in and ORCA is executed again."""
    init_logging()

    geom = geom_loader("lib:benzene.xyz")
    calc = ORCA(
        keywords="b3lyp def2-svp tightscf",
        pal=1,
        retry_calc=1,
    )
    geom.set_calculator(calc)

    def run_orca():
        geom.forces  # Execute ORCA by requesting an engrad calculation
        return "forces returned"

    def kill_orca():
        secs = 5
        sleep(secs)
        print(f"Waited {secs} s ")
        pid = check_output(["pidof", "orca"], text=True)
        pid = int(pid)
        kill(pid, SIGTERM)
        print("Killed orca")
        return "kill returned"

    # Somehow this seems unreliable ... sometimes it works, sometimes not.
    # Don't know why ...
    # Increasing the sleep time from 2 s to 5 s seems to help.
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in (run_orca, kill_orca)]
        for running_task in running_tasks:
            print(running_task.result())

    # Run ORCA here and kill it manually with `pkill orca`. This seems quite
    # reliable.
    # run_orca()
