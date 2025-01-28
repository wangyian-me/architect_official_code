import os
from build_dataset import BlenderkitDataset, large_furniture_list

# 1. Create your dataset
data_dir = os.environ.get("DATASET_PATH")
image_dir = os.path.join(data_dir, "rendered_images")

if not os.path.exists(image_dir):
    os.mkdir(image_dir)

# glb_dir is typically the same as data_dir if you download .glb files there
glb_dir = data_dir

ds = BlenderkitDataset(data_dir)

# ds.test_orientation(image_dir, glb_dir)

# 2. Loop through categories

for cat in large_furniture_list:
    print(f"=== Retrieving category: {cat} ===")
    # 3. Search & download up to 20 matches
    uids = ds.online_retrieve(cat)

    # 4. Annotate each newly downloaded file
    for uid in uids:
        print(f"Annotating: {uid}")
        # currently all are not large, set is_large=False
        ds.annotate_glb(uid, image_dir, glb_dir, is_large=True)

ds.save_database()

ds.encode_large_and_small_objects()
ds.encode_dino_and_clip_features_for_large_and_small_objects(image_dir)

print("Done.")
