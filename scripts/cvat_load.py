import os

import fiftyone as fo


DATASET_NAME = "BBM08S_head"
ANNO_KEY = "cvat_test_001"

CVAT_URL = "http://192.168.101.118:8080"


def main():
    dataset = fo.load_dataset(DATASET_NAME)

    print("Annotation runs:")
    print(dataset.list_annotation_runs())

    # 从 CVAT 下载标注并合并回原来的 ground_truth
    dataset.load_annotations(
        ANNO_KEY,
        url=CVAT_URL,
        username=os.environ["FIFTYONE_CVAT_USERNAME"],
        password=os.environ["FIFTYONE_CVAT_PASSWORD"],
    )

    print("Annotations loaded successfully.")

    # 得到当初送去 CVAT 的那批数据
    view = dataset.load_annotation_view(ANNO_KEY)

    print(f"Annotated samples: {len(view)}")


if __name__ == "__main__":
    main()