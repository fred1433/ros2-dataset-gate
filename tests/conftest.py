import logging
import os
from pathlib import Path

import pytest

from dsgate.demo import run_demo


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """The full `make demo` story, trained in-process (the container run is the CI demo step)."""
    out = tmp_path_factory.mktemp("demo") / "out"
    result = run_demo(out, train="process", parallel=4, quiet=True)
    result["out"] = Path(out)
    return result


@pytest.fixture(scope="session")
def s3(tmp_path_factory):
    """A moto S3 server for unit tests, reached over HTTP like the workers do."""
    from moto.server import ThreadedMotoServer

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    srv = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    srv.start()
    host, port = srv.get_host_and_port()
    env = {"DSGATE_S3_ENDPOINT": f"http://{host}:{port}", "AWS_ACCESS_KEY_ID": "testing",
           "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-east-1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    yield env
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    srv.stop()
