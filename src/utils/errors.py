"""Pipeline conditions that stop a run; ``main`` reports them without a traceback."""


class PipelineError(Exception):
    """Base class for errors that describe the data or configuration, not a bug."""


class MissingSourcesError(PipelineError):
    """A catalog source is neither complete nor blocked."""


class MissingInputError(PipelineError, FileNotFoundError):
    """A stage's input from an earlier stage does not exist; run that stage first."""


class ExportMismatchError(PipelineError, ValueError):
    """The tiktoken export does not reproduce ``tokenizer.json``'s ids."""


class PipelineBusyError(PipelineError):
    """Another run holds the lock on the same ``data/`` directory."""
