"""LocalMockScheduler for Adaptive Scheduler."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from adaptive_scheduler._scheduler.base_scheduler import BaseScheduler
from adaptive_scheduler._scheduler.common import run_submit

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from adaptive_scheduler.utils import EXECUTOR_TYPES


class LocalMockScheduler(BaseScheduler):
    """A scheduler that can be used for testing and runs locally."""

    # Attributes that all schedulers need to have
    _ext = ".batch"
    _JOB_ID_VARIABLE = "${JOB_ID}"

    def __init__(
        self,
        cores: int,
        *,
        python_executable: str | None = None,
        log_folder: str | Path = "",
        mpiexec_executable: str | None = None,
        executor_type: EXECUTOR_TYPES = "process-pool",
        num_threads: int = 1,
        extra_scheduler: list[str] | tuple[list[str], ...] | None = None,
        extra_env_vars: list[str] | None = None,
        extra_script: str | None = None,
        batch_folder: str | Path = "",
        # LocalMockScheduler specific
        mock_scheduler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the LocalMockScheduler."""
        import adaptive_scheduler._mock_scheduler

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
        # LocalMockScheduler specific
        self.mock_scheduler_kwargs = mock_scheduler_kwargs or {}
        self.mock_scheduler = adaptive_scheduler._mock_scheduler.MockScheduler(
            **self.mock_scheduler_kwargs,
        )
        mock_scheduler_file = adaptive_scheduler._mock_scheduler.__file__
        self.base_cmd = f"{self.python_executable} {mock_scheduler_file}"

        # Attributes that all schedulers need to have
        self._submit_cmd = f"{self.base_cmd} --submit"  # type: ignore[misc]
        self._cancel_cmd = f"{self.base_cmd} --cancel"  # type: ignore[misc]

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the scheduler."""
        # LocalMockScheduler has one different argument from the BaseScheduler
        return dict(
            **super().__getstate__(),
            mock_scheduler_kwargs=self.mock_scheduler_kwargs,
        )

    def job_script(self, options: dict[str, Any], *, index: int | None = None) -> str:
        """Get a jobscript in string form.

        Returns
        -------
        job_script
            A job script that can be submitted to PBS.
        index
            The index of the job that is being run. This is used when
            specifying different resources for different jobs.

        Notes
        -----
        Currently, there is a problem that this will not properly cleanup.
        for example `ipengine ... &` will be detached and go on,
        normally a scheduler will take care of this.

        """
        job_script = textwrap.dedent(
            """\
            #!/bin/sh

            {extra_env_vars}

            {extra_script}

            {executor_specific}
            """,
        )

        return job_script.format(
            extra_env_vars=self.extra_env_vars(index=index),
            executor_specific=self._executor_specific("${NAME}", options, index=index),
            extra_script=self.extra_script(index=index),
            job_id_variable=self._JOB_ID_VARIABLE,
        )

    def queue(self, *, me_only: bool = True) -> dict[str, dict]:  # noqa: ARG002
        """Get the queue of the scheduler."""
        return self.mock_scheduler.queue()

    def start_job(self, name: str, *, index: int | None = None) -> None:
        """Start a job."""
        if index is not None:
            msg = "LocalMockScheduler does not support `index`."
            raise NotImplementedError(msg)
        name_prefix = name.rsplit("-", 1)[0]
        submit_cmd = f"{self.submit_cmd} {name} {self.batch_fname(name_prefix)}"
        run_submit(submit_cmd, name)

    def extra_scheduler(self, *, index: int | None = None) -> str:  # noqa: ARG002
        """Get the extra scheduler options."""
        msg = "extra_scheduler is not implemented."
        raise NotImplementedError(msg)
