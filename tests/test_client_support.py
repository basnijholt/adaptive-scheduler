"""Tests for the `client_support` module."""

from __future__ import annotations

import tempfile
from unittest import mock

import adaptive
import pytest
import zmq

from adaptive_scheduler import client_support


@pytest.fixture(scope="module")
def zmq_url() -> str:
    """Create a ZMQ URL for testing purposes."""
    return "tcp://127.0.0.1:5555"


@pytest.fixture(scope="module")
def client(zmq_url: str) -> zmq.Socket:
    """Create a client for testing purposes."""
    ctx = zmq.Context()
    client = ctx.socket(zmq.REQ)
    client.connect(zmq_url)
    return client


@pytest.mark.asyncio()
async def test_get_learner(zmq_url: str) -> None:
    """Test `get_learner` function."""
    with tempfile.NamedTemporaryFile() as tmpfile:
        log_fname = tmpfile.name
        job_id = "test_job_id"
        job_name = "test_job_name"

        # Mock the fname_to_learner function to avoid creating real learners

        with mock.patch(
            "adaptive_scheduler.client_support.fname_to_learner",
        ) as mock_fname_to_learner:
            learner = adaptive.Learner1D(lambda x: x, (0, 1))
            initializer = None
            mock_fname_to_learner.return_value = (learner, initializer)
            # Patch `socket.recv_serialized` to return the desired reply
            with mock.patch(
                "zmq.sugar.socket.Socket.recv_serialized",
                return_value="Simple reply",
            ):
                learner, fname, initializer = client_support.get_learner(
                    zmq_url,
                    log_fname,
                    job_id,
                    job_name,
                )
                assert isinstance(learner, adaptive.Learner1D)
                assert isinstance(fname, str)
                assert initializer is None

            # Test no more learners available
            with mock.patch(
                "zmq.sugar.socket.Socket.recv_serialized",
                return_value=None,
            ), pytest.raises(RuntimeError, match="No learners to be run"):
                client_support.get_learner(
                    zmq_url,
                    log_fname,
                    job_id,
                    job_name,
                )

            # Test return exception
            with mock.patch(
                "adaptive_scheduler.client_support.log",
            ) as mock_log, mock.patch(
                "zmq.sugar.socket.Socket.recv_serialized",
                return_value=ValueError("Yo"),
            ):
                with pytest.raises(
                    ValueError,
                    match="Yo",
                ):
                    client_support.get_learner(
                        zmq_url,
                        log_fname,
                        job_id,
                        job_name,
                    )
                mock_log.exception.assert_called_with("got an exception")


@pytest.mark.asyncio()
async def test_tell_done(zmq_url: str) -> None:
    """Test `tell_done` function."""
    fname = "test_learner_file.pkl"
    with mock.patch("adaptive_scheduler.client_support.log") as mock_log, mock.patch(
        "zmq.sugar.socket.Socket.recv_serialized",
        return_value=None,
    ):
        client_support.tell_done(zmq_url, fname)
        mock_log.info.assert_called_with(
            "sent stop signal, going to wait 300s for a reply",
            fname="test_learner_file.pkl",
        )
