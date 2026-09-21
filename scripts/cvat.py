import os

import fiftyone as fo


DATASET_NAME = "BBM08S_head"
LABEL_FIELD = "ground_truth"

CVAT_URL = "http://192.168.101.118:8080"

ANNO_KEY = "cvat_test_001"
PROJECT_NAME = "BBM08S_head_test"


def main():
    # --------------------------------------------------
    # 1. 加载 FiftyOne Master Dataset
    # --------------------------------------------------
    dataset = fo.load_dataset(DATASET_NAME)

    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")

    # --------------------------------------------------
    # 2. 固定选择 10 张测试图片
    #
    # seed 很重要：
    # 如果以后重新执行，可以得到相同的随机结果
    # --------------------------------------------------
    view = dataset.take(10, seed=51)

    print(f"Selected samples: {len(view)}")

    for filepath in view.values("filepath"):
        print(filepath)

    # --------------------------------------------------
    # 3. 防止重复创建同名 annotation run
    # --------------------------------------------------
    if ANNO_KEY in dataset.list_annotation_runs():
        raise RuntimeError(
            f"Annotation run '{ANNO_KEY}' already exists.\n"
            "Please use another ANNO_KEY or delete the old run."
        )

    # --------------------------------------------------
    # 4. 发送到 CVAT
    #
    # ground_truth 已经存在：
    # 图片 + 现有 bbox 都会发送给 CVAT
    # --------------------------------------------------
    view.annotate(
        ANNO_KEY,

        backend="cvat",

        label_field=LABEL_FIELD,

        url=CVAT_URL,

        username=os.environ["FIFTYONE_CVAT_USERNAME"],
        password=os.environ["FIFTYONE_CVAT_PASSWORD"],

        project_name=PROJECT_NAME,

        # 第一轮测试只有10张
        # 一个job即可
        segment_size=10,

        # 服务器没有本地图形浏览器，因此不自动打开网页
        launch_editor=False,
    )

    # --------------------------------------------------
    # 5. 查看 FiftyOne 保存的 annotation run 信息
    # --------------------------------------------------
    print("\nCVAT task created successfully.")
    print(f"Annotation key: {ANNO_KEY}")

    info = dataset.get_annotation_info(ANNO_KEY)
    print(info)

    print("\nOpen CVAT:")
    print(CVAT_URL)


if __name__ == "__main__":
    main()