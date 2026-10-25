from typing import Mapping

from .runnable import IRunnable


def main(runnable: IRunnable, context: Mapping[str, object]) -> None:
    runnable.run(context)
