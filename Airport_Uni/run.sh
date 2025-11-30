#  Airport 데이터셋에 대해 단변량 시계열 예측을 수행하는 스크립트입니다.

# nested cross validation 아래와 같이 실행
## train/val/test: 2002~2015/2016/2017, 2002~2016/2017/2018, 2002~2017/2018/2019 --> 평균 내어 최적 파라미터로 2002~2018/2019/2024~2025(코로나 2020~2023 제외) 데이터셋에 적용

# 단변량 하이퍼파라미터 서치
## experiment setting
### lr: [1e-3, 1e-4, 1e-5]

## model setting
### seq_len: [28, 91, 365]
### pred_len: [1, 7]
### d_model: [64, 128, 256, 512]
### n_layers: [1, 2, 3, 4, 5]
### if patch base model(patchtst):
    # patch_len
    # if seq_len == [28]
        # patch_len: [7, 14]
    # else
        # patch_len: [7, 14, 28]
    
    # n_heads
    # if d_model == [64]
        # n_heads: [1, 2, 4, 8]
    # else
        # n_heads: [1, 2, 4, 8, 16]


# 단변량 예측 코드
cd Airport
python run.py --target arrival --gpu 0
python run.py --target departure --gpu 0