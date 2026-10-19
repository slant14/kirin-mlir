"""Out-of-tree MLIR interoperability for Kirin.

See ``docs/spec.md`` for the plan, ``docs/adr/0001-mlir-interop.md`` for the
boundary this is built on, and ``issues/`` for the ordered work items.

Layout, in dependency order:

- ``errors.py`` -- ``MLIRExportError`` and ``MLIRImportError``, the refusals
  out and in (issues 02, 06)
- ``parse.py``  -- MLIR generic form read back into a plain tree, the first
  stage of the import (issue 06)
- ``values.py`` -- Python payloads to builtin MLIR attributes and back
  (issues 02, 06)
- ``types.py``  -- Kirin type lattice to the ``!kirin.*`` family and back
  (issues 02, 06)
- ``attrs.py``  -- Kirin attributes to MLIR attributes and back (issues 02, 06)
- ``names.py``  -- Kirin names mangled into legal MLIR names (ADR 0002, issue 03)
- ``prepare.py`` -- export preconditions: typed, closure-free, one module (issue 03)
- ``emit.py``   -- the ``emit.mlir`` emitter over generic assembly form (issue 03)
- ``dialects/`` -- per-dialect ``emit.mlir`` tables, only Kirin ``func`` (issue 03)
- ``import_.py`` -- the parsed tree back into Kirin statements, blocks,
  regions and kernels, the second stage of the import (issue 06)
- ``irdl.py``   -- IRDL generation from Kirin ``decl`` metadata (issue 04)
- ``discover.py`` -- finding dialects by name, safely (issue 04)
- ``cli.py``    -- the ``kirin-mlir`` command line (issue 04)
"""

from . import dialects as dialects  # registers the per-dialect emit.mlir tables
from .emit import (
    EmitMLIR as EmitMLIR,
    MLIRFrame as MLIRFrame,
    emit_mlir as emit_mlir,
)
from .irdl import (
    Exclusion as Exclusion,
    GeneratedIRDL as GeneratedIRDL,
    exclusion as exclusion,
    generate_irdl as generate_irdl,
)
from .attrs import (
    emit_pyattr as emit_pyattr,
    emit_attribute as emit_attribute,
    register_attribute as register_attribute,
    emit_statement_attribute as emit_statement_attribute,
    emit_statement_attributes as emit_statement_attributes,
    register_attribute_import as register_attribute_import,
    import_statement_attribute as import_statement_attribute,
    import_statement_attributes as import_statement_attributes,
)
from .names import PREFIX as PREFIX, NameTable as NameTable, mangle as mangle
from .parse import (
    parse_type as parse_type,
    parse_module as parse_module,
    parse_attribute as parse_attribute,
)
from .types import (
    DIALECT as DIALECT,
    TYPE_SCHEMA as TYPE_SCHEMA,
    TYPE_HANDLERS as TYPE_HANDLERS,
    TYPE_IMPORTERS as TYPE_IMPORTERS,
    TypeSchema as TypeSchema,
    emit_type as emit_type,
    import_type as import_type,
    pyclass_type as pyclass_type,
    pyclass_named as pyclass_named,
    qualified_name as qualified_name,
)
from .errors import (
    MLIRExportError as MLIRExportError,
    MLIRImportError as MLIRImportError,
)
from .values import (
    escape_string as escape_string,
    emit_python_value as emit_python_value,
    import_python_value as import_python_value,
)
from .import_ import (
    SELF_NAME as SELF_NAME,
    MLIRImporter as MLIRImporter,
    ImportedModule as ImportedModule,
    import_mlir as import_mlir,
    register_statement_import as register_statement_import,
)
from .prepare import (
    PreparedModule as PreparedModule,
    prepare_for_mlir as prepare_for_mlir,
)
from .discover import find_dialects as find_dialects

__version__ = "0.1.0-dev"
