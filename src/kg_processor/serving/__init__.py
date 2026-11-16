# SPDX-License-Identifier: Apache-2.0
"""HTTP services that put a priority-aware enforcement floor in front of a fleet.

The serving plane is deliberately separate from the graph pipeline. Its modules
speak wire formats owned by other projects — the OpenAI chat contract and
MinerU's ``/file_parse`` contract — and exist so that priority, authentication,
and admission control hold no matter which engine or parser sits behind them.
"""
