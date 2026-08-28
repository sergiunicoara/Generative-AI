# Coverage report

- Total posts encountered: 50 feed headings; 51 provisional top-level feed records parsed from the final accessible snapshot.
- Unique posts: 51 provisional records; LinkedIn did not expose stable activity IDs in the UI.
- Posts successfully expanded: expansion attempted on every visible `see more` control; some lower-feed controls remained after the browser timeout and are marked `expanded: false`.
- Posts that could not be expanded: at least 8 in the retained snapshot; exact count is provisional because LinkedIn virtualizes cards.
- Relevant technical posts: 5 scored records in `relevant_posts.jsonl`, with additional lower-confidence observations in `graph_rag_findings.md`.
- External technical sources opened/verified: 6 primary/official sources in `sources.jsonl`.
- Oldest accessible post: 2yr.
- Newest post: 1w.
- Scroll iterations: 11, including 3 end-of-feed verification attempts.
- Termination reason: three additional End/scroll attempts produced no new posts and no “Show more results” control.

Extraction note: feed ordinals 14, 15, 34, and 35 were absent from the retained snapshot and are not fabricated. Scope statement: exhaustive across the posts exposed to the authenticated LinkedIn session during traversal. This is not a claim about posts hidden by membership permissions, ranking, virtualization, deletion, or LinkedIn server-side limits.
