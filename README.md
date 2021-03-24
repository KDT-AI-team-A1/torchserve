# TorchServe

PyTorch용 모델 서비스 프레임워크인 TorchServe를 이용하여 딥러닝 모델 배포

## Models

[detectron2](https://github.com/facebookresearch/detectron2)를 이용

[**Detectron2 Model Zoo**](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
- Faster R-CNN
- Cascade R-CNN

두 모델을 이용하여 마스크 인식 데이터셋 학습 후 inference 진행

## Requirements

📌 mar 파일 생성을 위해 필요한 파일
- 모델 state_dict() 파일 `.pth`
- 모델 config 파일 `.yaml`
- 모델 handler 파일 `.py`

1. `torch-model-archiver` 를 이용하여 mar 파일 생성

2. 생성된 mar 파일 model_store로 이동

3. 서버 실행 (배포를 위해 config.properties 추가)


## REST API

inference address: http://3.36.90.232:8080 

### Healthy Check API

*curl example*

```bash
curl http://3.36.90.232:8080/ping
```

- 정상적으로 작동중인 경우
  ```
  {
    "health": "healthy!"
  }
  ```


### Predictions API

[이미지](https://drive.google.com/file/d/1hloK0I8az-VXb56dYBMnp0adaP8Lv948/view?usp=sharing)를 다운받아 prediction test 진행
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hloK0I8az-VXb56dYBMnp0adaP8Lv948' -O input_mask.jpg
```

*curl example*

Faster R-CNN
```bash
curl http://3.36.90.232:8080/predictions/fastrcnn -T input_mask.jpg
```

Cascade R-CNN
```bash
curl http://3.36.90.232:8080/predictions/cascadercnn -T input_mask.jpg
```
