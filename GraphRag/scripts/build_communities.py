"""Offline: run Leiden community detection + Gemini summarization on full graph."""

import asyncio
import argparse

from graphrag.graph.community_builder import CommunityBuilder
from graphrag.graph.community_summarizer import CommunitySummarizer
from graphrag.graph.neo4j_client import get_neo4j


async def main(levels: int = 3):
    print(f"Building communities (levels={levels})...")
    builder = CommunityBuilder()
    communities = await builder.build()
    print(f"Found {len(communities)} communities. Summarizing...")

    summarizer = CommunitySummarizer()
    communities = await summarizer.summarize_all(communities)

    neo4j = get_neo4j()
    for c in communities:
        await neo4j.merge_community(c)

    await neo4j.close()
    print(f"Done. {len(communities)} communities written to Neo4j.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(levels=args.levels))
