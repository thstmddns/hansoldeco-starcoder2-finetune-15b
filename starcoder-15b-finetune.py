import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm
from sentence_transformers import SentenceTransformer 
from peft import LoraConfig


# cuda 사용여부 확인
# https://pytorch.org/get-started/locally/

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')


# data preprocessing
# 데이터 로드
data = pd.read_csv('data/train.csv')
quantization_config = BitsAndBytesConfig
compute_dtype = getattr(torch, 'float16')

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False)
checkpoint = 'bigcode/starcoder2-15b'


# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(checkpoint, eos_token='</s>')

# 데이터 포맷팅 및 토크나이징
formatted_data = []
for _, row in tqdm(data.iterrows()):
    for q_col in ['질문_1', '질문_2']:
        for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            #질문과 답변 쌍을 </s> token으로 연결
            input_text = row[q_col] + tokenizer.eos_token + row[a_col]
            input_ids = tokenizer.encode(input_text, return_tensors= 'pt')
            formatted_data.append(input_ids)
print('Done.')

# 모델로드
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config = quantization_config)
model.resize_token_embeddings(len(tokenizer)).to(device)

model.config.use_usecache = False

# 모델 학습 하이퍼파라미터(Hyperparameter) 세팅
# 실제 필요에 따라 조정
CFG = {
    'LR' : 2e-5, # Learning Rate
    'EPOCHS' : 10 # 학습 Epoch
}

# 모델 학습 설정
optimizer = AdamW(model.parameters(), lr=CFG['LR'])
model.train()

# 모델 학습
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    progress_bar = tqdm(enumerate(formatted_data), total=len(formatted_data))
    for batch_idx, batch in progress_bar:
        # 데이터를 gpu 단으로 이동
        batch = batch.to(device)
        outputs = model(batch, labels = batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 진행률 표시줄에 평균 손실 업데이트
        progress_bar.set_description(f"Epoch {epoch+1} - Avg Loss: {total_loss / (batch_idx+1):.4f}")

    # 에폭의 평균 손실을 출력
    print(f"Epoch {epoch+1}/{CFG['EPOCHS']}, Average Loss: {total_loss / len(formatted_data)}")

# 모델 저장
model.save_pretrained("./hansoldeco-starcoder-15b")
tokenizer.save_pretrained("./hansoldeco-starcoder-15b")

# 저장된 Fine-tuned 모델과 토크나이저 불러오기
model_dir = './hansoldeco-starcoder-15b'
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Inference를 위한 test.csv 파일로드
test = pd.read_csv('data/test.csv')

# test.csv의 '질문'에 대한 '답변'을 저장할 리스트
preds = []

# '질문' 컬럼의 각 질문에 대해 답변 생성
for test_question in tqdm(test['질문']):
    # 입력 텍슽트를 토큰화하고 모델 입력 형태로 변환
    input_ids = tokenizer.encode(test_question + tokenizer.eos_token, return_tensors='pt')

    # 답변 생성
    outputs_sequence = model.generate(
        input_ids = input_ids.to(device),
        max_length = 300,
        temperature = 0.9,
        top_k = 1,
        top_p = 0.9,
        repetition_penalty = 1.2,
        do_sample = True,
        num_return_sequences = 1
    )

    # 생성된 텍스트(답변) 저장
    for generated_sequence in outputs_sequence:
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        # 질문과 답변의 사이를 나타내는 eos_token (</s>)를 찾아, 이후부터 출력
        answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
        answer_only = full_text[answer_start:].strip().replace('\n', ' ')
        # 답변을 출력해보자
        # print(answer_only)
        preds.append(answer_only)

# Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
pred_embeddings = model.encode(preds)
# pred_embeddings.shape

submit = pd.read_csv('data/sample_submission.csv')
# 제출 양식 파일(sample_submission.csv)을 활용하여 embedding Vector로 변환한 결과를 삽입
submit.iloc[:, 1:] = pred_embeddings

#  리더보드 제출을 위한 csv 파일 생성
submit.to_csv(f'data/baseline_submit.csv', index=False) 