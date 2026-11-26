import uuid

class AnchorIndex:
    def __init__(self, client, collection, embedder, dimensions):
        from qdrant_client import models
        self.client, self.collection, self.embedder = client, collection, embedder
        self.models = models
        if not client.collection_exists(collection):
            client.create_collection(collection_name=collection, vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE))

    def _filter(self, parent_id=None, user_id=None, exclude_version=None):
        m = self.models
        must = []
        if parent_id is not None: must.append(m.FieldCondition(key='parent_id', match=m.MatchValue(value=str(parent_id))))
        if user_id is not None: must.append(m.FieldCondition(key='user_id', match=m.MatchValue(value=user_id)))
        must_not = [m.FieldCondition(key='version', match=m.MatchValue(value=exclude_version))] if exclude_version is not None else []
        return m.Filter(must=must, must_not=must_not)

    def replace(self, parent_id, user_id, version, anchors):
        m = self.models
        # Deterministic ids make a retry idempotent. Delete the old version only after upserting the new one.
        points = []
        for i, anchor in enumerate(anchors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{user_id}\0{parent_id}\0{version}\0{i}\0{anchor}'))
            points.append(m.PointStruct(id=point_id, vector=self.embedder.embed(anchor), payload={'parent_id':str(parent_id),'user_id':user_id,'version':version,'anchor':anchor}))
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        # Delete all previous points, including same-version points no longer in the desired list.
        existing, offset = [], None
        while True:
            page, offset = self.client.scroll(collection_name=self.collection, scroll_filter=self._filter(parent_id, user_id), limit=256, offset=offset, with_payload=False, with_vectors=False)
            existing.extend(str(p.id) for p in page)
            if offset is None: break
        keep = {p.id for p in points}
        stale = [p for p in existing if p not in keep]
        if stale:
            self.client.delete(collection_name=self.collection, points_selector=m.PointIdsList(points=stale), wait=True)

    def search(self, query, user_id, limit):
        hits = self.client.query_points(collection_name=self.collection, query=self.embedder.embed(query, query=True), query_filter=self._filter(user_id=user_id), limit=limit, with_payload=True).points
        return [{'parent_id':p.payload['parent_id'], 'version':p.payload['version'], 'score':p.score} for p in hits]
