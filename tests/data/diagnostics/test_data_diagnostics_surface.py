"""Identity and purity behavior floors for the data diagnostics surface.

The public five-block diagnostics surface must be lazily exposed by
``portlearn.data`` as exactly one additional name, compute with stdlib
imports only (never pandas/numpy at runtime, never instants or
availability knowledge), return immutable stdlib reports, add no
decision-time admission path, and behave as single-dataset pure
functions.  Each floor imports the diagnostics package inside the test
body so its absence reports as the causal failure of that floor.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from _synthetic import (
    FORBIDDEN_IMPORT_ROOTS_ANYWHERE,
    FORBIDDEN_RUNTIME_IMPORT_ROOTS,
    PACKAGE_ROOT,
    REPOSITORY_ROOT,
    any_scope_runtime_imports,
    first_field_name,
    import_diagnostics,
    instant,
    module_scope_runtime_imports,
    month_key,
    package_py_files,
    period_record,
    require_diagnostics_package,
    stdlib_modules,
    unqualified_dataset,
    walk_values,
)

LAZY_IMPORT_PROBE = """\
import sys

for name in [n for n in list(sys.modules) if n.startswith("portlearn")]:
    del sys.modules[name]

import portlearn.data

registered = {n for n in sys.modules if n.startswith("portlearn")}
assert "portlearn.data" in registered, f"portlearn.data not imported: {registered}"
assert "portlearn.data.dataset" not in sys.modules, (
    "importing portlearn.data eagerly imported the dataset module"
)
assert "portlearn.data.fama_french" not in sys.modules
assert "portlearn.data.fred" not in sys.modules
assert "pandas" not in sys.modules, (
    "importing portlearn.data eagerly imported pandas"
)

diagnostics = portlearn.data.diagnostics
assert diagnostics is not None, "portlearn.data.diagnostics did not resolve lazily"
assert "portlearn.data.diagnostics" in sys.modules
assert "portlearn.data.dataset" not in sys.modules, (
    "resolving portlearn.data.diagnostics eagerly imported the dataset module"
)
assert "pandas" not in sys.modules, (
    "resolving portlearn.data.diagnostics eagerly imported pandas"
)
for act in ("describe", "correlation", "coverage", "missingness", "plot"):
    assert callable(getattr(diagnostics, act, None)), (
        f"the diagnostics surface must expose a callable {act!r}"
    )
assert "portlearn.data.fama_french" not in sys.modules
assert "portlearn.data.fred" not in sys.modules
print("lazy-diagnostics: ok")
"""


def test_portlearn_data_lazily_exposes_diagnostics_as_one_additional_name() -> None:
    """``import portlearn.data`` registers that package alone; the
    diagnostics module resolves lazily afterwards; ``__all__`` is exactly
    the three prior public names plus ``diagnostics``."""
    import portlearn.data as data_package

    expected = {"ResearchDataset", "fama_french", "fred", "diagnostics"}
    assert sorted(data_package.__all__) == sorted(expected), (
        "portlearn.data.__all__ must gain exactly one additional public "
        f"name 'diagnostics'; expected {sorted(expected)}, found "
        f"{sorted(data_package.__all__)}"
    )
    assert len(data_package.__all__) == 4
    assert "diagnostics" in dir(data_package), (
        "the lazy attribute surface (__dir__/__getattr__) must expose 'diagnostics'"
    )

    environment = {
        key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"
    }
    completed = subprocess.run(
        [sys.executable, "-c", LAZY_IMPORT_PROBE],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "the lazy exposure law failed in a fresh interpreter:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


def test_diagnostics_computation_imports_neither_pandas_nor_numpy() -> None:
    """No runtime pandas/numpy import anywhere in the diagnostics package;
    module-scope runtime imports are stdlib only (a TYPE_CHECKING-only
    structural reference to the dataset module is type-only and
    permitted); the sole-pandas-home law survives verbatim."""
    require_diagnostics_package()

    for path in package_py_files():
        for fact in any_scope_runtime_imports(path):
            assert fact.root not in FORBIDDEN_RUNTIME_IMPORT_ROOTS, (
                f"{path.name} runtime-imports {fact.root!r} at "
                f"{fact.scope} scope: diagnostics computation is stdlib-only"
            )
            assert fact.root not in FORBIDDEN_IMPORT_ROOTS_ANYWHERE, (
                f"{path.name} imports {fact.root!r}: diagnostics are pure "
                "functions with no network/clock/randomness/filesystem "
                "surface"
            )
        for fact in module_scope_runtime_imports(path):
            assert fact.root in stdlib_modules(), (
                f"{path.name} has a module-scope runtime import of "
                f"{fact.root!r}: module scope must import stdlib only "
                "(structural dataset references stay TYPE_CHECKING-only)"
            )

    # The sole-pandas-home law: module-scope `import pandas` still lives in
    # portlearn/data/dataset.py alone across the whole package.
    pandas_homes = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if not path.is_file():
            continue
        for fact in module_scope_runtime_imports(path):
            if fact.root == "pandas":
                pandas_homes.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert pandas_homes == ["src/portlearn/data/dataset.py"], (
        "the sole pandas home must remain portlearn/data/dataset.py alone; "
        f"found module-scope pandas imports at {pandas_homes}"
    )


def test_diagnostics_constructs_no_instants_or_availability() -> None:
    """The diagnostics package never constructs instants, timezones, or
    availability policies, and never reads ``available_time``; no
    provider branch or provider literal exists anywhere in it."""
    require_diagnostics_package()

    banned_tokens = (
        "AvailabilityPolicy(",
        "ZoneInfo",
        "available_time",
        "timezone(",
        "astimezone(",
        "to_instant(",
        '"fred"',
        "'fred'",
        '"fama',
        "'fama",
        "provider ==",
        "provider==",
    )
    for path in package_py_files():
        text = path.read_text(encoding="utf-8")
        for token in banned_tokens:
            assert token not in text, (
                f"{path.name} contains the banned token {token!r}: "
                "diagnostics construct no instants, read no availability, "
                "and carry no provider-specific branch"
            )


def test_diagnostics_outputs_are_immutable_stdlib_types() -> None:
    """Every report/spec is a fixed stdlib structure: attribute mutation
    raises, no mutable dict/set surface exists, no pandas/numpy object
    appears, and value-equal datasets yield value-equal reports."""
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, month), value)
            for month, value in enumerate((2.0, 4.0, 9.0, 16.0), start=1)
        )
        + (period_record("beta", month_key(2023, 1), 7.0),)
    )
    equal_dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, month), value)
            for month, value in enumerate((2.0, 4.0, 9.0, 16.0), start=1)
        )
        + (period_record("beta", month_key(2023, 1), 7.0),)
    )
    grid = [("alpha", month_key(2023, m)) for m in (1, 2, 3, 4)] + [
        ("beta", month_key(2023, 1))
    ]

    outputs = [
        diagnostics.describe(dataset),
        diagnostics.correlation(dataset),
        diagnostics.coverage(dataset),
        diagnostics.missingness(dataset, expected_keys=grid),
        diagnostics.plot(dataset),
        diagnostics.plot(dataset, kind="describe"),
        diagnostics.plot(diagnostics.describe(dataset), kind="describe"),
    ]

    import dataclasses

    for report in outputs:
        assert dataclasses.is_dataclass(report), (
            f"every output must be a typed dataclass value; got {type(report).__name__}"
        )
        attribute = first_field_name(report)
        with pytest.raises((AttributeError, TypeError)):
            setattr(report, attribute, 1)
        for banned in (dict, set):
            for item in walk_values(report):
                assert not isinstance(item, banned), (
                    f"a mutable {banned.__name__} appears inside a report"
                )
        for item in walk_values(report):
            module = type(item).__module__
            assert not module.startswith(("pandas", "numpy")), (
                f"a {type(item).__name__} from {module!r} crossed the "
                "diagnostics boundary"
            )

    assert diagnostics.describe(dataset) == diagnostics.describe(equal_dataset)
    assert diagnostics.correlation(dataset) == diagnostics.correlation(equal_dataset)
    assert diagnostics.coverage(dataset) == diagnostics.coverage(equal_dataset)
    assert diagnostics.missingness(
        dataset, expected_keys=grid
    ) == diagnostics.missingness(equal_dataset, expected_keys=grid)
    assert diagnostics.plot(dataset) == diagnostics.plot(equal_dataset)


def test_diagnostics_adds_no_decision_time_admission_path() -> None:
    """No diagnostics output is accepted by a decision-time surface; an
    UNQUALIFIED dataset still refuses ``to_information_set``; the
    ResearchDataset public verb set is unchanged (no convenience methods)."""
    diagnostics = import_diagnostics()

    from portlearn.alignment import ObservationStore
    from portlearn.data.dataset import (
        QualifiedDataset,
        ResearchDataset,
        UnqualifiedDataError,
        UnqualifiedDataset,
    )
    from portlearn.interfaces import InformationSet
    from portlearn.transforms import RollingMean

    dataset = unqualified_dataset((period_record("alpha", month_key(2023, 1), 2.0),))
    with pytest.raises(UnqualifiedDataError):
        dataset.to_information_set()

    report = diagnostics.describe(dataset)
    for item in walk_values(report):
        assert not hasattr(item, "available_time"), (
            "diagnostics outputs carry no availability facts to admit"
        )

    decision = instant(2026, 1, 1)
    with pytest.raises((TypeError, AttributeError, ValueError)):
        InformationSet([report], as_of=decision)
    with pytest.raises((TypeError, AttributeError, ValueError)):
        RollingMean(2).transform([report])
    with pytest.raises((TypeError, AttributeError, ValueError)):
        ObservationStore([report])

    frozen_verbs = {
        "adapter_version",
        "availability",
        "availability_state",
        "auxiliary_bytes",
        "data_mode",
        "frequency",
        "name",
        "provider",
        "provider_dataset_id",
        "provenance",
        "qualify",
        "records",
        "retrieval_provenance",
        "source_bytes",
        "source_sha256",
        "to_information_set",
        "to_pandas",
        "units",
        "qualified_provenance",
    }
    for state_cls in (ResearchDataset, UnqualifiedDataset, QualifiedDataset):
        live = {name for name in dir(state_cls) if not name.startswith("_")}
        assert live == frozen_verbs, (
            f"{state_cls.__name__} public surface changed: no diagnostics "
            f"convenience method may exist; unexpected "
            f"{sorted(live - frozen_verbs)}, missing "
            f"{sorted(frozen_verbs - live)}"
        )


def test_diagnostics_are_single_dataset_pure_functions() -> None:
    """Cross-dataset/repeated-dataset inputs reject TypeError; repeated
    calls return value-equal reports; the package never imports a
    network/clock/randomness/filesystem surface."""
    diagnostics = import_diagnostics()

    dataset = unqualified_dataset(
        tuple(
            period_record("alpha", month_key(2023, month), float(month))
            for month in (1, 2, 3, 4)
        )
        + tuple(
            period_record("beta", month_key(2023, month), float(5 - month))
            for month in (1, 2, 3, 4)
        )
    )
    other = unqualified_dataset((period_record("gamma", month_key(2023, 1), 1.0),))
    grid = [("alpha", month_key(2023, m)) for m in (1, 2, 3, 4)]

    with pytest.raises(TypeError):
        diagnostics.describe(dataset, other)
    with pytest.raises(TypeError):
        diagnostics.correlation(dataset, other)
    with pytest.raises(TypeError):
        diagnostics.coverage(dataset, other)
    with pytest.raises(TypeError):
        diagnostics.plot(dataset, other)
    with pytest.raises(TypeError):
        diagnostics.missingness(dataset, other)

    assert diagnostics.describe(dataset) == diagnostics.describe(dataset)
    assert diagnostics.correlation(dataset) == diagnostics.correlation(dataset)
    assert diagnostics.coverage(dataset) == diagnostics.coverage(dataset)
    assert diagnostics.plot(dataset) == diagnostics.plot(dataset)
    assert diagnostics.missingness(
        dataset, expected_keys=grid
    ) == diagnostics.missingness(dataset, expected_keys=list(grid))

    require_diagnostics_package()
    for path in package_py_files():
        for fact in any_scope_runtime_imports(path):
            assert fact.root not in FORBIDDEN_IMPORT_ROOTS_ANYWHERE, (
                f"{path.name} imports {fact.root!r}: diagnostics must be "
                "pure (no network, clock, randomness, filesystem, or "
                "subprocess surface)"
            )


def test_diagnostics_package_present_for_source_scans() -> None:
    """Guard floor companion: the package directory must exist so every
    source-scan floor scans real implementation bytes, never a vacuous
    absent path."""
    require_diagnostics_package()
    assert package_py_files(), (
        "the diagnostics package exists but declares no Python modules"
    )
