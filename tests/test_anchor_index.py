from casp_mem.integration.anchor_index import AnchorIndex

class Embed:
    def embed(self, text, query=False): return [1.0, 0.0] if 'tea' in text else [0.0,1.0]

def test_anchor_replace_and_user_filter_in_temporary_qdrant():
    from qdrant_client import QdrantClient
    index = AnchorIndex(QdrantClient(':memory:'), 'anchors', Embed(), 2)
    index.replace('parent', 'alice', 1, ['tea'])
    assert index.search('tea', 'alice', 5)[0]['version'] == 1
    assert index.search('tea', 'bob', 5) == []
    index.replace('parent', 'alice', 2, ['coffee'])
    hits = index.search('coffee', 'alice', 5)
    assert len(hits) == 1 and hits[0]['version'] == 2
    index.replace('parent', 'alice', 0, [])
    assert index.search('coffee', 'alice', 5) == []
