"""
Regression test for graph_ego_walk strict mode silent bypass.

Bug: When GRAPH_HL_KEYWORD_MODE=strict but hl_keywords is empty/None,
the edge filter was silently bypassed (no drop) but log still reported
hl_mode=strict, creating misleading observability.

Fix: emit a warning log when strict mode is set but hl_keyword_set is empty.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_kg_and_vdb():
    """Mock the knowledge_graph_inst and entities_vdb for _perform_graph_ego_walk."""
    kg = MagicMock()
    kg.get_nodes_edges_batch = AsyncMock(
        return_value={
            "seed_a": [("seed_a", "node_b")],
            "node_b": [("node_b", "node_c")],
        }
    )
    kg.get_edges_batch = AsyncMock(
        return_value={
            ("seed_a", "node_b"): {
                "description": "khoa cong nghe thong tin",
                "keywords": "cntt, khoa",
                "weight": 1.0,
                "file_path": "test.pdf",
                "source_id": "chunk-1",
            },
        }
    )
    kg.get_nodes_batch = AsyncMock(
        return_value={
            "seed_a": {"description": "seed", "entity_type": "ORG", "file_path": "", "source_id": ""},
            "node_b": {"description": "khoa CNTT", "entity_type": "ORG", "file_path": "", "source_id": ""},
        }
    )
    kg.node_degrees_batch = AsyncMock(return_value={"seed_a": 1, "node_b": 1})

    vdb = MagicMock()
    vdb.query = AsyncMock(
        return_value=[
            {"entity_name": "seed_a", "distance": 0.9},
        ]
    )

    return kg, vdb


@pytest.mark.asyncio
async def test_strict_mode_empty_hl_keywords_logs_warning(caplog, mock_kg_and_vdb):
    """
    When hl_mode=strict and hl_keywords is empty,
    a WARNING must be logged and edges must NOT be silently dropped.
    """
    import logging
    from lightrag.operate import _perform_graph_ego_walk

    kg, vdb = mock_kg_and_vdb

    entity_chunks = MagicMock()
    entity_chunks.get_by_ids = AsyncMock(return_value=[])
    relation_chunks = MagicMock()
    relation_chunks.get_by_ids = AsyncMock(return_value=[])

    query_param = MagicMock()
    query_param.top_k = 10

    lightrag_logger = logging.getLogger("lightrag")
    lightrag_logger.addHandler(caplog.handler)
    try:
        with patch.dict(os.environ, {"GRAPH_HL_KEYWORD_MODE": "strict"}):
            with caplog.at_level(logging.WARNING, logger="lightrag"):
                entities, relations = await _perform_graph_ego_walk(
                    query="TLU co khoa nao?",
                    ll_keywords="",
                    hl_keywords="",
                    entities_vdb=vdb,
                    knowledge_graph_inst=kg,
                    entity_chunks_storage=entity_chunks,
                    relation_chunks_storage=relation_chunks,
                    query_param=query_param,
                )
    finally:
        lightrag_logger.removeHandler(caplog.handler)

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("hl_mode=strict" in m and "hl_keywords empty" in m for m in warning_messages), (
        f"Expected warning about strict mode bypass, got: {warning_messages}"
    )
    assert len(entities) > 0
    assert len(relations) > 0


@pytest.mark.asyncio
async def test_strict_mode_with_hl_keywords_drops_non_matching(mock_kg_and_vdb):
    """
    When hl_mode=strict and hl_keywords is non-empty,
    edges with overlap=0 must be dropped (no warning).
    """
    import logging
    from lightrag.operate import _perform_graph_ego_walk

    kg, vdb = mock_kg_and_vdb

    entity_chunks = MagicMock()
    entity_chunks.get_by_ids = AsyncMock(return_value=[])
    relation_chunks = MagicMock()
    relation_chunks.get_by_ids = AsyncMock(return_value=[])

    query_param = MagicMock()
    query_param.top_k = 10

    with patch.dict(os.environ, {"GRAPH_HL_KEYWORD_MODE": "strict"}):
        with patch("logging.Logger.warning") as mock_warn:
            entities, relations = await _perform_graph_ego_walk(
                query="tim nganh y",
                ll_keywords="y, suc khoe",
                hl_keywords="y, suc khoe",
                entities_vdb=vdb,
                knowledge_graph_inst=kg,
                entity_chunks_storage=entity_chunks,
                relation_chunks_storage=relation_chunks,
                query_param=query_param,
            )

    warn_messages = [str(c.args) for c in mock_warn.call_args_list]
    assert not any("hl_mode=strict" in str(m) and "empty" in str(m) for m in warn_messages), (
        f"Should NOT warn when hl_keywords is provided, got: {warn_messages}"
    )
    assert relations == [], "Edge with overlap=0 must be dropped in strict mode"


@pytest.mark.asyncio
async def test_soft_mode_no_warning(mock_kg_and_vdb, caplog):
    """
    When hl_mode=soft (default), no warning should be emitted
    regardless of hl_keywords state.
    """
    import logging
    from lightrag.operate import _perform_graph_ego_walk

    kg, vdb = mock_kg_and_vdb

    entity_chunks = MagicMock()
    entity_chunks.get_by_ids = AsyncMock(return_value=[])
    relation_chunks = MagicMock()
    relation_chunks.get_by_ids = AsyncMock(return_value=[])

    query_param = MagicMock()
    query_param.top_k = 10

    lightrag_logger = logging.getLogger("lightrag")
    lightrag_logger.addHandler(caplog.handler)
    try:
        with patch.dict(os.environ, {"GRAPH_HL_KEYWORD_MODE": "soft"}):
            with caplog.at_level(logging.WARNING, logger="lightrag"):
                entities, relations = await _perform_graph_ego_walk(
                    query="test",
                    ll_keywords="",
                    hl_keywords="",
                    entities_vdb=vdb,
                    knowledge_graph_inst=kg,
                    entity_chunks_storage=entity_chunks,
                    relation_chunks_storage=relation_chunks,
                    query_param=query_param,
                )
    finally:
        lightrag_logger.removeHandler(caplog.handler)

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("hl_mode=strict" in m for m in warning_messages)
