# space-agent

## Active Research Demo Dataset

The Korean Streamlit demo uses the **GPT-5.4-nano v6** mapping results.
This dataset was confirmed as the research baseline on 2026-10-14.

- Mapping source: `outputs/openai_batch_full_gpt54nano_v6/placeness_mapping_sentences.csv`
- Deployment data: `demo_outputs/gpt54nano_v6/`
- Mapped reviews: 263,888 across 10,978 places
- Phrase-factor assignments: 648,463

The original `llm_factor_mapping_sentences_full.csv` contains older GPT-4o-mini
results and is retained only as an archive. The demo does not load it or fall
back to it. Restoring an older interface does not change the mapping dataset.

The interface separates factor mapping from scoring. Factor scores are the
arithmetic mean of phrase values: positive +1, neutral/mixed 0, negative -1.
Unmapped factors have no score. Mention proportions and weighted scores are
not used in the demo.

## Run and Verify

```powershell
python -m streamlit run app.py
python -m unittest discover -s tests -p test_research_demo.py
```

The tracked Parquet files are sufficient to run the deployed demo. To rebuild
them locally, the full mapping source CSV must be available:

```powershell
python scripts/build_demo_scoring_data.py
```

Rebuilding these files calculates scores from existing mappings; it does not
call an LLM or generate new factor assignments.
