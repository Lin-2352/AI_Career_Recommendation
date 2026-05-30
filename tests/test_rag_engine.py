from __future__ import annotations

import pandas as pd

from backend.chatbot.rag_engine import DatasetRAGEngine


def test_rag_engine_discovers_all_csv_files(tmp_path) -> None:
    """RAG discovery indexes every CSV in the raw data folder unless a limit is requested."""
    for index in range(3):
        pd.DataFrame({"skill": [f"python {index}"], "career": ["Data Analyst"]}).to_csv(
            tmp_path / f"dataset_{index}.csv",
            index=False,
        )
    engine = DatasetRAGEngine(raw_dir=tmp_path)

    assert len(engine.discover_dataset_files()) == 3
    assert len(engine.discover_dataset_files(limit=2)) == 2


def test_rag_engine_retrieves_relevant_rows(tmp_path) -> None:
    """RAG retrieval returns row chunks matching the user query."""
    pd.DataFrame(
        {
            "skill": ["python sql dashboards", "biology chemistry patient care"],
            "career": ["Data Analyst", "Doctor"],
        }
    ).to_csv(tmp_path / "careers.csv", index=False)
    engine = DatasetRAGEngine(raw_dir=tmp_path)
    engine.build_index()

    chunks = engine.retrieve("SQL dashboard career", top_k=1)

    assert len(chunks) == 1
    assert "Data Analyst" in chunks[0].text
