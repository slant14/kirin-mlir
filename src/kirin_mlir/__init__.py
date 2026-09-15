"""Out-of-tree MLIR interoperability for Kirin.

See ``docs/spec.md`` for the plan, ``docs/adr/0001-mlir-interop.md`` for the
boundary this is built on, and ``issues/`` for the ordered work items.

Layout, in dependency order:

- ``errors.py`` -- ``MLIRExportError``, the one refusal (issue 02)
- ``values.py`` -- Python payloads to builtin MLIR attributes (issue 02)
- ``types.py``  -- Kirin type lattice to the ``!kirin.*`` family (issue 02)
- ``attrs.py``  -- Kirin attributes to MLIR attributes (issue 02)
- ``prepare.py`` -- the ``PrepareForMLIR`` pass, export preconditions (issue 03)
- ``emit.py``   -- the ``emit.mlir`` emitter over generic assembly form (issue 03)
- ``irdl.py``   -- IRDL generation from Kirin ``decl`` metadata (issue 04)
"""

from .attrs import (
    emit_pyattr as emit_pyattr,
    emit_attribute as emit_attribute,
    register_attribute as register_attribute,
    emit_statement_attributes as emit_statement_attributes,
)
from .types import (
    DIALECT as DIALECT,
    TYPE_SCHEMA as TYPE_SCHEMA,
    TYPE_HANDLERS as TYPE_HANDLERS,
    TypeSchema as TypeSchema,
    emit_type as emit_type,
    pyclass_type as pyclass_type,
    qualified_name as qualified_name,
)
from .errors import MLIRExportError as MLIRExportError
from .values import (
    escape_string as escape_string,
    emit_python_value as emit_python_value,
)

__version__ = "0.1.0-dev"
