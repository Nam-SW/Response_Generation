# Response_Generation
 implement NLP paper.
 
[Learning a Simple and Effective Model for Multi-turn Response Generation with Auxiliary Tasks]  
![image1]  


## 유의사항
 + 학습에 사용되는 데이터는 1:1 채팅 데이터이며, 다음과 같은 형식으로 utf-8 인코딩으로 저장되어야 합니다.
   |         time        | writer |    contents   |
   |:-------------------:|:------:|:-------------:|
   | 2020-01-01 00:01:00 | user1  | 새해다 새해   |
   | 2020-01-01 00:01:00 | user1  | 성인 어서오고 |
   | 2020-01-01 00:02:00 | user2  | 늙었네ㅋㅋ    |

 + `remove_names.json`은 데이터에서 가릴 이름들을 저장한 리스트입니다.
   이메일, 전화번호 등과 달리 이름은 모두 가리기가 어려우므로, 사용자가 직접 가릴 이름등을 설정할 수 있습니다.
   다수의 문자열을 포함한 리스트가 .json 형식으로 저장되어야 합니다.

## Tokenizer
 + remove_names: remove_names.json 의 경로
 + data_dir: 기본적인 raw 데이터들이 저장된 디렉토리 경로
 + save_dir: tokenizer를 저장할 경로
```bash
$ python3 run_tokenizer_training.py \
  --limit_alphabet 6000 \
  --vocab_size 8000 \
  --remove_names remove_names.json \
  --data_dir DATA DIRECTORY \
  --save_dir config
```

## Preprocessing
 + raw_data_dir: raw 데이터들이 저장된 디렉토리 경로
 + preprocessed_data_dir: 중간 저장 데이터들이 저장될 경로
 + model_use_data_dir: 모델이 사용할 데이터가 저장될 경로
 + remove_names: remove_names.json 의 경로
 + tokenizer_config: tokenizer.json의 경로
 + utterance_size: 모델이 한 번 예측할 때 사용할 이전 발화의 개수
 + max_len: 각 발화의 최대 길이 (토큰 기준)
 + use_multiprocessing: 멀티프로세싱을 사용 여부
```bash
$ python3 run_setup_data.py \
  --raw_data_dir RAW_DATA_DIRECTROY \
  --preprocessed_data_dir INTERMEDIATE_PREPROCESSING_DATA_DIRECTROY \
  --model_use_data_dir SAVE_DIRECTROY \
  --remove_names remove_names.json \
  --tokenizer_config tokenizer.json \
  --utterance_size 4 \
  --max_len 100 \
  --use_multiprocessing False
```

## Training
 + data_dir: Preprocessing 단계에서 생성한 model_use_data 경로
 + validation_split: train/validation split 비율
 + data_shuffle: 데이터 섞기
 + model_save_dir: 모델을 저장할 경로
 + tensorboard_log_dir: tensorboard 로그를 저장할 경로. 미입력시 로그 저장 x
 + learning_rate: 학습률
 + batch_size: batch size
 + global_max_step: 학습할 총 step
 + validation_step: 몇 step마다 저장과 검증을 진행할지 
 + verbose: cnffur duqn. 0 or 1
 + gpu_count: 사용할 gpu 개수
 + model_hparams: model_config.json의 경로
 + tokenizer: 검증할 때 테스트로 예측해볼 때 사용할 tokenizer.json의 경로. 미입력시 테스트 예측 x
 + load_latest: 최근 check point를 불러올지 여부
```bash
$ python3 run_model_training.py \
  --data_dir data/model_use_data \
  --validation_split 0.1 \
  --data_shuffle True \
  --model_save_dir model \
  --tensorboard_log_dir logs \
  --learning_rate 0.05 \
  --batch_size 64 \
  --global_max_step 50000 \
  --validation_step 1000 \
  --verbose 1 \
  --gpu_count 2 \
  --model_hparams config/model_hparams.json \
  --tokenizer config/tokenizer.json \
  --load_latest False
```

[Learning a Simple and Effective Model for Multi-turn Response Generation with Auxiliary Tasks]: https://arxiv.org/abs/2004.01972
[image1]: https://blog.pingpong.us/images/2020.11.11.emnlp2020-preview/model-structure-with-auxiliary-tasks.png
[image2]: https://blog.pingpong.us/images/2020.11.11.emnlp2020-preview/auxiliary-tasks.png
