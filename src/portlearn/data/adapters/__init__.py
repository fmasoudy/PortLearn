"""Provider-format adapters for external research datasets.

This package hosts the thin provider-format adapters (fetch/decode split): each adapter is a pure decoder over locally-held bytes that
reduces a provider's publication format to the ingestion declared-table
form, plus a separate stdlib-only fetch function whose only outputs are
bytes plus retrieval provenance.

Decoded datasets are research inputs and are **not investable**; the
package's adapters never claim investability for any dataset.
"""
