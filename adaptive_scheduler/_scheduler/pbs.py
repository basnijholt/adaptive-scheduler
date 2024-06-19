"""PBS for Adaptive Scheduler."""

from __future__ import annotations

import collections
import getpass
import math
import os
import os.path
import subprocess
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler
from adaptive_scheduler._scheduler.common import run_submit

if TYPE_CHECKING:
    from typing import Any

    from adaptive_scheduler.utils import EXECUTOR_TYPES


class PBS(BaseScheduler):
    """PBS scheduler."""

    # Attributes that all schedulers need to have
    _ext = ".batch"
    # the "-k oe" flags with "qsub" writes the log output to
    # files directly instead of at the end of the job. The downside
    # is that the logfiles are put in the homefolder.
    _submit_cmd = "qsub -k oe"
    _JOB_ID_VARIABLE = "${PBS_JOBID}"
    _options_flag = "PBS"
    _cancel_cmd = "qdel"

    def __init__(
        self,
        cores: int,
        *,
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: EXECUTOR_TYPES = "process-pool",
        num_threads: int = 1,
        extra_scheduler: list[str] | None = None,
        extra_env_vars: list[str] | None = None,
        extra_script: str | None = None,
        batch_folder: str | Path = "",
        # Extra PBS specific arguments
        cores_per_node: int | None = None,
    ) -> None:
        """Initialize the PBS scheduler."""
        super().__init__(
            cores,
            python_executable=python_executable,
            log_folder=log_folder,
            mpiexec_executable=mpiexec_executable,
            executor_type=executor_type,
            num_threads=num_threads,
            extra_scheduler=extra_scheduler,
            extra_env_vars=extra_env_vars,
            extra_script=extra_script,
            batch_folder=batch_folder,
        )
        # PBS specific
        self.cores_per_node = cores_per_node
        self._calculate_nnodes()
        if cores != self.cores:
            warnings.warn(
                f"`self.cores` changed from {cores} to {self.cores}",
                stacklevel=2,
            )

    def __getstate__(self) -> dict[str, Any]:
        """Return the state of the scheduler."""
        # PBS has one different argument from the BaseScheduler
        return dict(**super().__getstate__(), cores_per_node=self.cores_per_node)

    @staticmethod
    def sanatize_job_id(job_id: str) -> str:
        """Changes '91722.hpc05.hpc' into '91722'."""
        return job_id.split(".")[0]

    def _calculate_nnodes(self) -> None:
        assert isinstance(self.cores, int), "self.cores must be an integer for PBS."
        if self.cores_per_node is None:
            partial_msg = "Use set `cores_per_node=...` before passing the scheduler."
            try:
                max_cores_per_node = self._guess_cores_per_node()
                self.nnodes = math.ceil(self.cores / max_cores_per_node)
                self.cores_per_node = round(self.cores / self.nnodes)
                msg = (
                    f"`#PBS -l nodes={self.nnodes}:ppn={self.cores_per_node}` is"
                    f" guessed using the `qnodes` command, we set"
                    f" `cores_per_node={self.cores_per_node}`."
                    f" You might want to change this. {partial_msg}"
                )
                warnings.warn(msg, stacklevel=2)
                self.cores = self.nnodes * self.cores_per_node
            except Exception as e:  # noqa: BLE001
                msg = (
                    f"Got an error: {e}."
                    " Couldn't guess `cores_per_node`, this argument is required"
                    f" for PBS. {partial_msg}"
                    " We set `cores_per_node=1`!"
                )
                warnings.warn(msg, stacklevel=2)
                self.nnodes = self.cores
                self.cores_per_nodes = 1
        else:
            self.nnodes = self.cores / self.cores_per_node  # type: ignore[assignment]
            if not float(self.nnodes).is_integer():
                msg = "cores / cores_per_node must be an integer!"
                raise ValueError(msg)
            self.nnodes = int(self.nnodes)

    def output_fnames(self, name: str) -> list[Path]:
        """Get the output filenames."""
        # The "-k oe" flags with "qsub" writes the log output to
        # files directly instead of at the end of the job. The downside
        # is that the logfiles are put in the home folder.
        home = Path.home()
        stdout, stderr = (home / f"{name}.{x}{self._JOB_ID_VARIABLE}" for x in "oe")
        return [stdout, stderr]

    def job_script(self, options: dict[str, Any], *, index: int | None = None) -> str:
        """Get a jobscript in string form.

        Returns
        -------
        job_script
            A job script that can be submitted to PBS.
        index
            The index of the job that is being run. This is used when
            specifying different resources for different jobs.
            Currently not implemented for PBS!

        """
        job_script = textwrap.dedent(
            f"""\
            #!/bin/sh
            #PBS -l nodes={self.nnodes}:ppn={self.cores_per_node}
            #PBS -V
            #PBS -o /tmp/placeholder
            #PBS -e /tmp/placeholder
            {{extra_scheduler}}

            {{extra_env_vars}}

            cd $PBS_O_WORKDIR

            {{extra_script}}

            {{executor_specific}}
            """,
        )

        return job_script.format(
            extra_scheduler=self.extra_scheduler(index=index),
            extra_env_vars=self.extra_env_vars(index=index),
            extra_script=self.extra_script(index=index),
            executor_specific=self._executor_specific("${NAME}", options, index=index),
            job_id_variable=self._JOB_ID_VARIABLE,
        )

    def start_job(self, name: str, *, index: int | None = None) -> None:
        """Writes a job script and submits it to the scheduler."""
        if index is not None:
            msg = "PBS does not support `index`."
            raise NotImplementedError(msg)
        name_prefix = name.rsplit("-", 1)[0]
        name_opt = f"-N {name}"
        submit_cmd = f"{self.submit_cmd} {name_opt} {self.batch_fname(name_prefix)}"
        run_submit(submit_cmd, name)

    @staticmethod
    def _split_by_job(lines: list[str]) -> list[list[str]]:
        jobs: list[list[str]] = [[]]
        for line in lines:
            line = line.strip()  # noqa: PLW2901
            if line:
                jobs[-1].append(line)
            else:
                jobs.append([])
        return [j for j in jobs if j]

    @staticmethod
    def _fix_line_cuts(raw_info: list[str]) -> list[str]:
        info = []
        for line in raw_info:
            if " = " in line:
                info.append(line)
            else:
                info[-1] += line
        return info

    def queue(self, *, me_only: bool = True) -> dict[str, dict]:
        """Get the status of all jobs in the queue."""
        cmd = ["qstat", "-f"]

        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            env=dict(os.environ, SGE_LONG_QNAMES="1000"),
        )
        output = proc.stdout

        if proc.returncode != 0:
            msg = "qstat is not responding."
            raise RuntimeError(msg)

        jobs = self._split_by_job(output.replace("\n\t", "").split("\n"))

        running = {}
        for header, *raw_info in jobs:
            job_id = header.split("Job Id: ")[1]
            info = dict([line.split(" = ") for line in self._fix_line_cuts(raw_info)])
            if info["job_state"] in ["R", "Q"]:
                info["job_name"] = info["Job_Name"]  # used in `server_support.manage_jobs`
                info["state"] = info["job_state"]  # used in `RunManager.live`
                running[job_id] = info

        if me_only:
            # We do this because the "-u [username here]"  flag doesn't
            # work with "-f" on some clusters.
            username = getpass.getuser()
            running = {
                job_id: info for job_id, info in running.items() if username in info["Job_Owner"]
            }

        return running

    def _qnodes(self) -> dict[str, dict[str, str]]:
        proc = subprocess.run(["qnodes"], text=True, capture_output=True, check=False)
        output = proc.stdout

        if proc.returncode != 0:
            msg = "qnodes is not responding."
            raise RuntimeError(msg)

        jobs = self._split_by_job(output.replace("\n\t", "").split("\n"))

        return {
            node: dict([line.split(" = ") for line in self._fix_line_cuts(raw_info)])
            for node, *raw_info in jobs
        }

    def _guess_cores_per_node(self) -> int:
        nodes = self._qnodes()
        cntr = collections.Counter([int(info["np"]) for info in nodes.values()])
        ncores, freq = cntr.most_common(1)[0]
        return ncores
