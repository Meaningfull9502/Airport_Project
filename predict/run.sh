cd ./predict
python predict_real.py \
    --target both \
    --model itransformer

python predict_real.py \
    --target both \
    --model lstm

python predict_real.py \
    --target both \
    --model tsmixer_covar