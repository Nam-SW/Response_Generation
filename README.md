# Response_Generation
 implement NLP paper.
 
[Learning a Simple and Effective Model for Multi-turn Response Generation with Auxiliary Tasks]  
![image1]  
![image2]  


## 유의사항
 + Raw Data는 다음과 같은 형식으로, utf-8 인코딩으로 저장되어야 합니다.
   |         time        | writer |    contents   |
   |:-------------------:|:------:|:-------------:|
   | 2020-01-01 00:01:00 | user1  | 새해다 새해   |
   | 2020-01-01 00:01:00 | user1  | 성인 어서오고 |
   | 2020-01-01 00:02:00 | user2  | 늙었네ㅋㅋ    |

 + `remove_names.json`은 데이터에서 가릴 이름들을 저장한 리스트입니다.
   이메일, 전화번호 등과 달리 이름은 모두 가리기가 어려우므로, 사용자가 직접 가릴 이름등을 설정할 수 있습니다.
   다수의 문자열을 포함한 리스트가 .json 형식으로 저장되어야 합니다.

## Tokenizer
```bash
$ python3 run_tokenizer_training.py \
  --limit_alphabet 6000 \
  --vocab_size 8000 \
  --remove_names remove_names.json \
  --data_dir DATA DIRECTORY \
  --save_dir config
```

## Preprocessing
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
```bash
$ python3 run_model_training.py \
  --data_dir DATA_DIRECTORY \
  --validation_split 0.1 \
  --data_shuffle True \
  --model_save_dir model \
  --learning_rate 0.0001 \
  --batch_size 80 \
  --epochs 5 \
  --verbose 1 \
  --model_hparams config/model_hparams.json
```

[Learning a Simple and Effective Model for Multi-turn Response Generation with Auxiliary Tasks]: https://arxiv.org/abs/2004.01972
[image1]: https://blog.pingpong.us/images/2020.11.11.emnlp2020-preview/model-structure-with-auxiliary-tasks.png
[image2]: https://blog.pingpong.us/images/2020.11.11.emnlp2020-preview/auxiliary-tasks.png
