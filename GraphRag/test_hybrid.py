import asyncio
from graphrag.retrieval.local_search import LocalSearch


async def main():
    ls = LocalSearch()

    # Cross-document query: ownership.txt links Elon Musk → SpaceX
    # products.txt + achievements.txt have SpaceX rockets
    result = await ls.search("What rockets did Elon Musk's company launch?")

    chunks = result["chunks"]
    entities = result["entities"]

    print(f"Chunks returned  : {len(chunks)}")
    print(f"Entities returned: {len(entities)}")
    print()

    for c in chunks:
        tag = c.get("retrieval", "?")
        score = c.get("score", 0)
        via = c.get("via_entity", "")
        text = c["text"][:90]
        via_str = f"  via={via}" if via else ""
        print(f"  [{tag:12s}] score={score:.4f}{via_str}")
        print(f"    {text}")
        print()


asyncio.run(main())
