DATASET_FOLDER="./data/celeba_hq"
# raw images to LMDB format
TARGET_SIZE=256,1024
for DATASET_TYPE in "train" "test" "val"; do
    python preprocessor/prepare_data.py --out $DATASET_FOLDER/LMDB_$DATASET_TYPE --size $TARGET_SIZE $DATASET_FOLDER/raw_images/$DATASET_TYPE
done