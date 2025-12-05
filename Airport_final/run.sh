#  Airport 데이터셋에 대해 시계열 예측을 수행하는 스크립트입니다.
# nested cross validation 아래와 같이 실행
## train/val/test: 2002~2015/2016/2017, 2002~2016/2017/2018, 2002~2017/2018/2019 --> 평균 내어 최적 파라미터로 2002~2018/2019/2024~2025(코로나 2020~2023 제외) 데이터셋에 적용

# 하이퍼파라미터 서치
## experiment setting
### lr: [1e-3, 1e-4, 1e-5]

## model setting
### seq_len: [56, 91, 365]
### d_model: [64, 128, 256, 512]
### n_layers: [2, 4, 6]
### if patch based model(patchtst, timexer):
    # patch_len: [7, 14]
### if transformer model(patchtst, itransformer, timexer): 
    # n_heads
    # if d_model == [64]
        # n_heads: [2, 4, 8]
    # else
        # n_heads: [2, 4, 8, 16]


#  예측 코드
## target: arrival, departure, both
## pred_len: 14, 56
## if both --> use_covar(미래 예측에 공변량 사용 여부): True, False
## if arrival, departure --> use_covar(과거 임베딩 및 미래 예측에 공변량 사용 여부): True, False / But Not NBeats

# 학습
# 공변량 사용
cd ./Airport
python run.py --target both --pred_len 56 --use_covar
python run.py --target both --pred_len 14 --use_covar

python run.py --target arrival --pred_len 56 --use_covar
python run.py --target arrival --pred_len 56 
python run.py --target arrival --pred_len 14 --use_covar
python run.py --target arrival --pred_len 14 

python run.py --target departure --pred_len 56 --use_covar
python run.py --target departure --pred_len 56 
python run.py --target departure --pred_len 14 --use_covar
python run.py --target departure --pred_len 14

python run.py --target both --pred_len 56 --use_covar --use_dbloss
python run.py --target arrival --pred_len 56 --use_covar --use_dbloss
python run.py --target departure --pred_len 56 --use_covar --use_dbloss
