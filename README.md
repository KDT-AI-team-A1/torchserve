# TorchServe

## Windows 환경

### Prerequisites

> [https://github.com/pytorch/serve/blob/master/docs/torchserve_on_win_native.md#prerequisites](https://github.com/pytorch/serve/blob/master/docs/torchserve_on_win_native.md#prerequisites)

- 필요 파일 설치
  - Git
  - openjdk11

## Object Detection 예제
> [https://github.com/pytorch/serve/tree/master/examples/object_detector/fast-rcnn](https://github.com/pytorch/serve/tree/master/examples/object_detector/fast-rcnn)

### mar 파일 생성

1. [model.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) 파일 다운로드 후 `torch-model-archiver` 를 이용하여 파일 생성

```bash
torch-model-archiver --model-name fastrcnn --version 1.0 --model-file examples/object_detector/fast-rcnn/model.py --serialized-file fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --handler object_detector --extra-files examples/object_detector/index_to_name.json
```

2. 생성된 mar파일 model_store로 이동
```bash
mkdir model_store
mv fastrcnn.mar model_store/
```

### 서버 실행

```bash
torchserve --start --model-store model_store --models fastrcnn=fastrcnn.mar --ncs
```

### 테스트

```bash
curl http://127.0.0.1:8080/predictions/fastrcnn -T examples/object_detector/persons.jpg
```

### 서버 종료

```bash
torchserve --stop
```
