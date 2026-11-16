"""Submission-to-worker contracts for explicitly selected ontology vocabularies."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from test_distributed_execution import MemoryDistributedStore, _settings

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.application.distributed_planner import DistributedRunPlanner
from kg_processor.application.distributed_worker import settings_for_run
from kg_processor.application.ontology import load_ontology
from kg_processor.domain.distributed import RevisionRequest, RunDefinition, RunStatus


@pytest.mark.parametrize("source", ["profile_path", "graph_labels"])
@pytest.mark.parametrize("revision", [False, True])
def test_planner_carries_selected_vocabulary_by_value(
    tmp_path: Path, source: str, revision: bool
) -> None:
    """Workers use the submitter's selection without reading its profile path."""

    (tmp_path / "document.txt").write_text("Synthetic public document.")
    settings = _settings(tmp_path)
    if source == "profile_path":
        profile_path = tmp_path / "ontology.yaml"
        profile_path.write_text(
            yaml.safe_dump(
                {
                    "name": "selected",
                    "description": "Synthetic vocabulary.",
                    "entity_types": [{"name": "CUSTOM_TYPE", "description": "A custom type."}],
                    "relation_types": [{"name": "CUSTOM_LINK", "description": "A custom link."}],
                }
            )
        )
        settings = settings.model_copy(
            update={"ontology": settings.ontology.model_copy(update={"profile_path": profile_path})}
        )
    else:
        settings = settings.model_copy(
            update={
                "graph": settings.graph.model_copy(
                    update={"entity_types": ["CUSTOM_TYPE"], "relation_types": ["CUSTOM_LINK"]}
                )
            }
        )
    store = MemoryDistributedStore()
    if revision:
        store.earlier_runs["base-run"] = (
            RunDefinition(
                id="base-run",
                graph_id=settings.job.graph_id,
                config={},
                config_digest="base",
                status=RunStatus.SUCCEEDED,
            ),
            [],
        )
    summary = DistributedRunPlanner(
        settings, LocalFileSource(tmp_path, ["*.txt"]), store, store
    ).submit(
        run_id="selected-run",
        revision=RevisionRequest(base_run_id="base-run") if revision else None,
    )
    if source == "profile_path":
        profile_path.unlink()
    worker = settings.model_copy(
        update={
            "ontology": settings.ontology.model_copy(
                update={"profile_path": None, "profile": None}
            ),
            "graph": settings.graph.model_copy(
                update={"entity_types": ["FLEET_TYPE"], "relation_types": ["FLEET_LINK"]}
            ),
        }
    )
    effective = settings_for_run(worker, summary.run.config, run_id=summary.run.id)
    selected = load_ontology(
        effective.ontology.profile_path,
        effective.graph.entity_types,
        effective.graph.relation_types,
        inline=effective.ontology.profile,
    ).profile
    assert [value.name for value in selected.entity_types] == ["CUSTOM_TYPE"]
    assert [value.name for value in selected.relation_types] == ["CUSTOM_LINK"]
    assert effective.ontology.profile_path is None
    assert effective.job.job_id == "selected-run"


def test_omitted_ontology_retains_the_mounted_worker_default(tmp_path: Path) -> None:
    """An absent selection leaves vocabulary choice to the configured fleet."""

    (tmp_path / "document.txt").write_text("Synthetic public document.")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    summary = DistributedRunPlanner(
        settings, LocalFileSource(tmp_path, ["*.txt"]), store, store
    ).submit()
    assert summary.run.config["ontology"] == {"profile_path": None, "profile": None}
    mounted = settings.ontology.model_copy(update={"profile_path": tmp_path / "mounted.yaml"})
    worker = settings.model_copy(update={"ontology": mounted})
    assert settings_for_run(worker, summary.run.config).ontology is mounted
