import hydra
from transformers import AutoModelForCausalLM, GPT2TokenizerFast


@hydra.main(config_name="config.yaml")
def main(cfg):
    # tokenizer 로드
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.PATH.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(cfg.PATH.output_dir).cuda()

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    sep = tokenizer.sep_token

    talk_log = []

    while True:
        # 유저 입력단
        user_input = ""
        while True:
            utterance = input("me: ")
            if not utterance:
                break
            elif utterance == "종료":
                exit()

            user_input += (f" {sep} " if user_input else "") + utterance

        talk_log.append(f"{bos}{user_input}{eos}")

        # 모델 예측
        input_text = "".join(talk_log[-4:] if talk_log else "") + bos
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
        input_len = input_ids.shape[-1]

        output = model.generate(
            input_ids,
            max_length=256,
            # do_sample=True,
            top_k=10,
            top_p=0.95,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=0,
        )[0, input_len - 1 :].cpu()

        talk_log.append(tokenizer.decode(output))
        model_output = tokenizer.decode(output[1:-1])

        # 모델 결과 출력
        for utterance in model_output.split(" <sep> "):
            print(f"AI: {utterance}")


if __name__ == "__main__":
    main()
