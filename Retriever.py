import json
import time
import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import (
            AutoProcessor as ClipProcessor, 
            CLIPModel, 
            AutoImageProcessor as DinoProcessor, 
            AutoModel
        )

class BlenderkitRetriever():
    def __init__(self, data_dir, image_method="dino"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.data_dir = data_dir
        self.json_dir = os.path.join(self.data_dir, "blenderkit_database.json")
        
        if not os.path.exists(self.json_dir):
            # Initialize an empty JSON if it doesn't exist
            with open(self.json_dir, "w") as f:
                json.dump({}, f)
                
        # Load the database from JSON
        with open(self.json_dir, "r") as f:
            self.database = json.load(f)

        # 1) Load text embeddings (SBERT) for large objects
        large_data = torch.load(
            os.path.join(self.data_dir, "descriptions_embeddings_large.pt"),
            map_location=device
        )
        self.large_ids = large_data["ids"]  # list of UIDs
        self.descriptions_embeddings_large = large_data["embeddings"].to(device)

        # 2) Load text embeddings (SBERT) for small objects
        small_data = torch.load(
            os.path.join(self.data_dir, "descriptions_embeddings_small.pt"),
            map_location=device
        )
        self.small_ids = small_data["ids"]
        self.descriptions_embeddings_small = small_data["embeddings"].to(device)

        # 3) Load the SBERT model
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2').to(device)

        # 4) Optionally load CLIP embeddings for large/small
        #    (If you have them saved, uncomment the lines below)
        if image_method == "clip":
            clip_large_path = os.path.join(self.data_dir, "clip_embeddings_large.pt")
            clip_small_path = os.path.join(self.data_dir, "clip_embeddings_small.pt")
            if os.path.exists(clip_large_path) and os.path.exists(clip_small_path):
                clip_large_data = torch.load(clip_large_path, map_location=device)
                self.clip_large_ids = clip_large_data["ids"]
                self.clip_embeddings_large = clip_large_data["embeddings"].to(device)

                clip_small_data = torch.load(clip_small_path, map_location=device)
                self.clip_small_ids = clip_small_data["ids"]
                self.clip_embeddings_small = clip_small_data["embeddings"].to(device)
            else:
                self.clip_large_ids = []
                self.clip_embeddings_large = None
                self.clip_small_ids = []
                self.clip_embeddings_small = None
            # If you will be encoding new images:
            self.clip_processor = ClipProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_model.eval()

        # 5) Optionally load DINO embeddings for large/small
        #    (If you have them saved, uncomment the lines below)
        if image_method == "dino":
            dino_large_path = os.path.join(self.data_dir, "dino_embeddings_large.pt")
            dino_small_path = os.path.join(self.data_dir, "dino_embeddings_small.pt")
            if os.path.exists(dino_large_path) and os.path.exists(dino_small_path):
                dino_large_data = torch.load(dino_large_path, map_location=device)
                self.dino_large_ids = dino_large_data["ids"]
                self.dino_embeddings_large = dino_large_data["embeddings"].to(device)

                dino_small_data = torch.load(dino_small_path, map_location=device)
                self.dino_small_ids = dino_small_data["ids"]
                self.dino_embeddings_small = dino_small_data["embeddings"].to(device)
            else:
                self.dino_large_ids = []
                self.dino_embeddings_large = None
                self.dino_small_ids = []
                self.dino_embeddings_small = None

            # 6) Optionally load the CLIP/DINO models if you want to
            #    encode a new query image on-the-fly.

            self.dino_processor = DinoProcessor.from_pretrained("facebook/dinov2-base")
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            self.dino_model.eval()
        
        self.device = device
        self.image_method = image_method

        # 7) A list of names to reject
        self.reject_list = ["rug", "carpet", "curtains", "curtain"]
        self.free = False

    def retrieve(self, 
                 obj_name, 
                 obj_description, 
                 reference_bbox, 
                 is_large=True):
        """
        Original text-based (SBERT) retrieval.

        Steps:
          1) Filter by name if there's any match.
          2) Encode obj_description (SBERT).
          3) Compute cosine similarity with large/small group embeddings.
          4) Take top-5 by similarity.
          5) Among these top-5, pick the bounding box that best matches reference_bbox.
          6) Return best_id or None.
        """
        # Name rejection
        if obj_name in self.reject_list:
            return None

        device = self.descriptions_embeddings_large.device  # same device

        # 1) Which group to query
        if is_large:
            group_ids = self.large_ids
            group_embeds = self.descriptions_embeddings_large
        else:
            group_ids = self.small_ids
            group_embeds = self.descriptions_embeddings_small

        if not group_ids:
            return None

        # 2) Filter by name if there's a match
        name_matched_indices = []
        for i, asset_id in enumerate(group_ids):
            if self.database.get(asset_id, {}).get("name", "") == obj_name:
                name_matched_indices.append(i)

        if len(name_matched_indices) > 0:
            filtered_indices = name_matched_indices
        else:
            filtered_indices = list(range(len(group_ids)))

        if not filtered_indices:
            return None

        # 3) Encode text query
        query_emb = self.sbert_model.encode(obj_description, convert_to_tensor=True).to(device)
        # 4) Compute similarities
        filtered_embeddings = group_embeds[filtered_indices]
        similarities = F.cosine_similarity(query_emb, filtered_embeddings, dim=1)

        # 5) top-5
        top_k = min(5, len(filtered_indices))
        top_vals, top_idxs = torch.topk(similarities, top_k)
        top_asset_indices = [filtered_indices[idx.item()] for idx in top_idxs]
        top_asset_ids = [group_ids[i] for i in top_asset_indices]

        # 6) bounding box pick
        best_id = self._select_best_bbox(top_asset_ids, reference_bbox, is_large)
        return best_id

    def _select_best_bbox(self, candidate_ids, reference_bbox, is_large):
        """
        Among candidate_ids, pick the bounding box that best matches reference_bbox.

        If is_large=True, we test two orientations (swap x,y).
        If is_large=False, we sort dimensions and compare.
        """
        def bbox_distance(ref_bbox, stored_bbox, is_large):
            if len(ref_bbox) != 3 or len(stored_bbox) != 3:
                raise ValueError("Both ref_bbox and stored_bbox must have exactly 3 dimensions.")
            rx, ry, rz = ref_bbox
            sx, sz, sy = stored_bbox

            if is_large:
                # orientation A
                dist_a = ((rx - sx)**2 + (ry - sy)**2 + (rz - sz)**2)**0.5
                # orientation B
                dist_b = ((rx - sy)**2 + (ry - sx)**2 + (rz - sz)**2)**0.5
                return min(dist_a, dist_b)
            else:
                # For smaller objects, sort dims and compare
                ref_sorted = sorted([rx, ry, rz])
                sto_sorted = sorted([sx, sy, sz])
                dist = sum((rs - ss)**2 for rs, ss in zip(ref_sorted, sto_sorted))
                return dist**0.5

        best_id = None
        best_dist = float('inf')
        for uid in candidate_ids:
            stored_bbox = self.database.get(uid, {}).get("bounding_box_extents", [0,0,0])
            dist = bbox_distance(reference_bbox, stored_bbox, is_large)
            if dist < best_dist:
                best_dist = dist
                best_id = uid
        
        return best_id
    
    def _fallback_text_only(self, filtered_indices, obj_description, reference_bbox, is_large, top_k):
        """
        If no items pass the image threshold, or there's no image query,
        fallback to a text-only approach on the subset 'filtered_indices'.
        We compute text similarity for those items, pick top_k, then bounding box.
        """
        import torch
        import torch.nn.functional as F

        device = self.device

        # If there's no text description, we can't do text retrieval
        if not obj_description:
            return None

        # Decide which group
        if is_large:
            text_ids = self.large_ids
            text_embeds = self.descriptions_embeddings_large
        else:
            text_ids = self.small_ids
            text_embeds = self.descriptions_embeddings_small

        # encode text query
        query_emb = self.sbert_model.encode(obj_description, convert_to_tensor=True).to(device)
        if query_emb.ndim == 1:
            query_emb = query_emb.unsqueeze(0)

        # subset the embeddings
        embeds_filtered = text_embeds[filtered_indices]
        # compute similarity
        sims = F.cosine_similarity(query_emb, embeds_filtered, dim=1)
        # top_k
        top_vals, top_idxs = torch.topk(sims, min(top_k, len(filtered_indices)))
        top_asset_indices = [filtered_indices[idx.item()] for idx in top_idxs]
        top_asset_ids = [text_ids[i] for i in top_asset_indices]

        # bounding box
        return self._select_best_bbox(top_asset_ids, reference_bbox, is_large)


    def retrieve_hybrid(
        self,
        obj_name=None,
        obj_description=None,
        reference_bbox=None,
        is_large=True,
        query_image_path=None,
        top_k=3,
        ratio=0.5
    ):
        """
        Hybrid retrieval with new pipeline logic:
        1) Compute image similarity for all items in the group.
        2) Keep items with image_sim > 0.45 => filtered set A.
        3) If A is not empty:
            For each item in A, also compute text similarity (SBERT).
            Combine them: combined_score = text_sim + image_sim.
            Sort descending, take top_k => bounding box.
            If A is empty:
            fallback => text-only:
                compute text_sim for entire group => pick top_k => bounding box.

        Arguments:
        - obj_name (str)        : optional name filter
        - obj_description (str) : text description for SBERT
        - reference_bbox (list) : [rx, ry, rz]
        - is_large (bool)       : pick large or small group
        - query_image_path (str): path to query image
        - top_k (int)           : how many final items to consider for bounding box
        - image_method (str)    : "clip" or "dino"

        Returns:
        (str or None) best_id
        """
        import os
        import torch
        import torch.nn.functional as F
        from PIL import Image

        image_method = self.image_method

        if reference_bbox is None:
            reference_bbox = [1, 1, 1]

        # 1) Decide which group to query
        if is_large:
            text_ids = self.large_ids
            text_embeds = self.descriptions_embeddings_large
            if image_method == "clip":
                image_ids = self.clip_large_ids
                image_embeds = self.clip_embeddings_large
            else:  # "dino"
                image_ids = self.dino_large_ids
                image_embeds = self.dino_embeddings_large
        else:
            text_ids = self.small_ids
            text_embeds = self.descriptions_embeddings_small
            if image_method == "clip":
                image_ids = self.clip_small_ids
                image_embeds = self.clip_embeddings_small
            else:  # "dino"
                image_ids = self.dino_small_ids
                image_embeds = self.dino_embeddings_small

        # If database is empty, nothing to retrieve
        if not text_ids:
            return None

        # Name filter (optional)
        if obj_name and obj_name in self.reject_list:
            return None

        filtered_indices = list(range(len(text_ids)))

        if not filtered_indices:
            return None

        # Make sure we have an image for the image-based step
        if (not query_image_path) or (not os.path.exists(query_image_path)):
            # If no query image => fallback to text-only
            return self._fallback_text_only(filtered_indices, obj_description, reference_bbox, is_large, top_k)

        # 2) Encode the query image
        device = self.device
        image = Image.open(query_image_path).convert("RGB")

        if image_method == "clip":
            if not image_ids or image_embeds is None:
                # No stored image embeddings => fallback text-only
                return self._fallback_text_only(filtered_indices, obj_description, reference_bbox, is_large, top_k)
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
                img_feats = self.clip_model.get_image_features(**inputs)
                img_feats = F.normalize(img_feats, dim=-1)  # (1, embed_dim)
        else:  # "dino"
            if not image_ids or image_embeds is None:
                return self._fallback_text_only(filtered_indices, obj_description, reference_bbox, is_large, top_k)
            with torch.no_grad():
                inputs = self.dino_processor(images=image, return_tensors="pt").to(device)
                outputs = self.dino_model(**inputs)
                img_feats = outputs.last_hidden_state.mean(dim=1)
                img_feats = F.normalize(img_feats, dim=-1)  # (1, hidden_dim)

        # 3) Image similarity for all items in `filtered_indices`
        # We need to map these indices to the corresponding row in image_embeds
        id2img_idx = {uid: i for i, uid in enumerate(image_ids)}
        image_sim_list = []
        for idx in filtered_indices:
            uid = text_ids[idx]
            img_idx = id2img_idx.get(uid, None)
            if img_idx is not None:
                db_img_emb = image_embeds[img_idx]  # shape (embed_dim,)
                sim = F.cosine_similarity(img_feats, db_img_emb.unsqueeze(0), dim=1).item()
            else:
                sim = 0.0
            image_sim_list.append((idx, sim))

        # Filter to those with sim > 0.45
        filtered_by_image = [(i, s) for (i, s) in image_sim_list if s > 0.45]

        # 4) If no items pass 0.45 => fallback to text-only
        if len(filtered_by_image) == 0:
            return self._fallback_text_only(filtered_indices, obj_description, reference_bbox, is_large, top_k)

        # 5) Among those that pass, compute text similarity (SBERT),
        #    and then combine (text_sim + image_sim)
        if not obj_description:
            obj_description = obj_name

        # Encode text query
        text_query_emb = self.sbert_model.encode(obj_description, convert_to_tensor=True).to(device)
        if text_query_emb.ndim == 1:
            text_query_emb = text_query_emb.unsqueeze(0)

        # We'll compute text similarity for these filtered items
        text_sims = []
        text_embeds_filtered = text_embeds[filtered_indices]  # shape (N, embed_dim)
        # But we only need it for items in 'filtered_by_image'
        for (item_idx, img_sim) in filtered_by_image:
            txt_emb = text_embeds_filtered[item_idx - filtered_indices[0]] if False else None
            # Actually we need to offset if item_idx is an absolute index? Let's do it carefully:
            # Actually simpler approach: direct indexing: text_embeds[item_idx]
            txt_emb = text_embeds[item_idx]  # shape (embed_dim,)
            txt_sim = F.cosine_similarity(text_query_emb, txt_emb.unsqueeze(0), dim=1).item()
            combined_score = ratio * txt_sim + (1 - ratio) * img_sim
            text_sims.append((item_idx, txt_sim, img_sim, combined_score))

        # 6) Sort by combined_score descending, take top_k
        text_sims.sort(key=lambda x: x[3], reverse=True)
        top_candidates = text_sims[:top_k]
        top_asset_ids = [text_ids[x[0]] for x in top_candidates]

        # 7) bounding box selection among top-k
        return self._select_best_bbox(top_asset_ids, reference_bbox, is_large)

        
    def get_image_text_topk(self, query_image_path, description, k, is_large=True):
        """
        Given a query image and a text description, compute:
        - Top-k text similarity (SBERT) in the chosen size group (large/small)
        - Top-k image similarity (CLIP or DINO) in the chosen size group

        Returns:
        (topk_text_ids, topk_img_ids)
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        # 1) Decide which group (large or small)
        if is_large:
            text_ids = self.large_ids
            text_embeds = self.descriptions_embeddings_large
            if self.image_method == "clip":
                image_ids = self.clip_large_ids
                image_embeds = self.clip_embeddings_large
            else:  # "dino"
                image_ids = self.dino_large_ids
                image_embeds = self.dino_embeddings_large
        else:
            text_ids = self.small_ids
            text_embeds = self.descriptions_embeddings_small
            if self.image_method == "clip":
                image_ids = self.clip_small_ids
                image_embeds = self.clip_embeddings_small
            else:  # "dino"
                image_ids = self.dino_small_ids
                image_embeds = self.dino_embeddings_small

        device = self.device

        # 2) Encode the query text with SBERT
        text_query_emb = self.sbert_model.encode(description, convert_to_tensor=True).to(device)
        # shape: (1, embed_dim) or (embed_dim,) => ensure it's (1, embed_dim)
        if len(text_query_emb.shape) == 1:
            text_query_emb = text_query_emb.unsqueeze(0)

        # 3) Compute text similarities for all items in this group
        # text_embeds shape: (num_items, embed_dim)
        text_sims = F.cosine_similarity(text_query_emb, text_embeds, dim=1)  # shape (num_items,)
        # get top-k
        topk_text_vals, topk_text_idxs = torch.topk(text_sims, k)
        topk_text_ids = [text_ids[idx.item()] for idx in topk_text_idxs]

        # 4) Encode the query image
        image = Image.open(query_image_path).convert("RGB")

        if self.image_method == "clip":
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
                img_feats = self.clip_model.get_image_features(**inputs)
                # Normalize so cosine similarity is just dot product
                img_feats = F.normalize(img_feats, dim=-1)
        else:  # "dino"
            with torch.no_grad():
                inputs = self.dino_processor(images=image, return_tensors="pt").to(device)
                outputs = self.dino_model(**inputs)
                # outputs.last_hidden_state: shape (1, seq_len, hidden_dim)
                img_feats = outputs.last_hidden_state.mean(dim=1)  # shape (1, hidden_dim)
                img_feats = F.normalize(img_feats, dim=-1)

        # 5) Compute image similarities
        # image_embeds shape: (num_items, embed_dim)
        # img_feats shape: (1, embed_dim)
        img_sims = F.cosine_similarity(img_feats, image_embeds, dim=1)  # shape (num_items,)
        topk_img_vals, topk_img_idxs = torch.topk(img_sims, k)
        topk_img_ids = [image_ids[idx.item()] for idx in topk_img_idxs]

        print("text vals:", topk_text_vals)
        print("image vals:", topk_img_vals)
        return topk_text_ids, topk_img_ids
