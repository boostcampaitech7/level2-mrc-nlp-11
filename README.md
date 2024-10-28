# Open Domain Question Answering

## 1 프로젝트 개요

**1.1 개요**

ODQA는 대규모의 문서 데이터(ex. wikipedia 문서, 뉴스 기사)에서 질문에 대한 답변을 찾는 과제이다. 일반적인 QA와 달리 질문의 답변을 찾을 문서가 미리 정해져 있지 않으므로, 적절한 문서를 찾는 과정(passage retrieval)까지 구축해야 한다는 점이 특징으로, 크게 두 단계를 거쳐 답변을 찾을 수 있다.

1. **Passage Retrieval - Retriever**
Passage Retrieval 단계에서 Retriever는 질문의 답변이 포함되어 있을 것으로 예상되는 문서를 검색한다. Sparse Embedding, Dense Embedding 등을 활용하여 질문과 문서의 유사도를 계산한 뒤, 그 중 유사도가 가장 높은 k개의 문서를 찾아 반환한다.
2. **MRC(Machine Reading Comprehension) - Reader**
MRC 단계에서는 Reader 모델이 Retriever가 찾은 문서들에서 질문의 답변에 해당하는 텍스트를 찾는다. 본 대회에서는 문서에서 답변에 해당하는 텍스트를 직접 추출하는 **Extraction-based QA**를 수행하므로, 답변의 시작 위치와, 끝 위치를 예측하도록 모델을 학습시킨다.

<img width="600" src="https://github.com/user-attachments/assets/fd591a7d-9c14-4597-8a7d-2811a42f1c61" />

<br />
<br />

**1.2 평가지표**

본 대회의 평가지표는 Exact Match(EM)으로, 전체 샘플 중 예측과 정답이 정확히 일치하는 샘플의 비율로 측정한다.

<br />

## 2 프로젝트 팀 구성 및 역할

## 팀원 소개

| **이름** | **프로필** | **역할** | **GitHub** |
| --- | --- | --- | --- |
| **강정완** | <img alt="강정완" width="140" height="140" src="https://github.com/user-attachments/assets/4f48f414-1da1-4476-acfa-b73104604db7" /> | - 프로젝트 템플릿 제작 <br /> - Retriever(Dense, Sparse, Re-ranking) <br /> - 데이터 후처리(조사 제거) | [GJ98](https://github.com/GJ98) |
| **김민선** | <img alt="김민선" width="140" height="140" src="https://github.com/user-attachments/assets/603a2aaa-58ea-416e-b366-097f845bf5d5" /> | - 프로젝트 협업 관리(깃허브 이슈 템플릿 및 pre-commit 설정, <br /> 노션 및 깃허브 이슈 관리) <br /> - 데이터 전처리(마크다운 전처리) <br /> - 데이터 증강(기존 미활용 문서 활용 새 질문 답변 생성) | [CLM-BONNY](https://github.com/CLM-BONNY) |
| **서선아** | <img alt="서선아" width="140" height="140" src="https://github.com/user-attachments/assets/57c9c737-28d7-4ed0-b8c9-48eb5daaeb8a" /> | - 데이터 전처리(title, context 병합) <br /> - 데이터 증강(기존 활용 문서 새 질문 답변 생성) | [seon03](https://github.com/seon03) |
| **이인구** | <img alt="이인구" width="140" height="140" src="https://github.com/user-attachments/assets/51a26c46-03e5-4a42-94de-9f7f3034383c" /> | - 데이터 전처리(질문 조사 제거, 명사 추출 병합) <br /> - 데이터 증강(ETRI_MRC) | [inguelee](https://github.com/inguelee) |
| **이재협** | <img alt="이재협" width="140" height="140" src="https://github.com/user-attachments/assets/75b8ef71-6afe-4654-9843-a0913937a1de" /> | - Streamlit 데이터 뷰어 제작, 데이터 전처리(위키피디아 문서 전처리) <br /> - 데이터 증강(기존 질문 수정) <br /> - Reader(Separate Inference) | [jhyeop](https://github.com/jhyeop) |
| **임상엽** | <img alt="임상엽" width="140" height="140" src="https://github.com/user-attachments/assets/2d66dd95-8c1c-4441-9511-6bf2fc3b06170" /> | - Disk Manager 구현 <br /> - Commitizen 설정 <br /> - 데이터 전처리(Question 타입 분류) <br /> - Reader(Transfer Learning) | [gityeop](https://github.com/gityeop) |

<br />

## 3 프로젝트

**3.1 프로젝트 진행 일정**

- EDA, 전처리, 증강, Reader, Retriever, 앙상블 순서로 진행
- 각 단계는 회의, 분담, 이슈생성, 작업(Commit, Push), Pull Request, Merge 순서로 진행

<img width="700" alt="project" src="https://github.com/user-attachments/assets/ced065f6-d20b-4a40-8925-a2bf3e255b27" />

<br />
<br />

**3.2 프로젝트 폴더 구조**

```
template
├── config/
|   ├── retrieval.py               # Retreival 설정 파일
|   └── mrc.py                     # MRC 설정 파일
├── module/
|   ├── loss.py                    # loss 함수 (torch)
|   ├── metric.py                  # metric 함수 (huggingface.evaluate)
|   ├── data.py                    # data 클래스 (pl.LightningDataModule)
|   ├── encoder.py                 # Retrieval 인코더 클래스 (AutoPretrainedModel)
|   ├── retrieval.py               # Retrieval 모델 클래스 (pl.LightningModule)
│   └── mrc.py                     # MRC 모델 클래스 (pl.LightningModule)
├── utils/                         
├── inference.py                   # validation/test dataset 추론 파일
├── train_mrc.py                   # MRC 모델 학습 파일
├── train_dense_retrieval.py       # dense retrieval 모델 학습 파일
└── train_sparse_retrieval.py      # sparse retrieval 모델 학습 파일
```

<br />

## 4 EDA

**4.1 위키피디아 문서별 토큰 개수의 분포**

<img width="450" src="https://github.com/user-attachments/assets/85527467-6513-422e-9b62-496fd299607b" />


- 위키피디아 문서를 `klue/bert-base` 토크나이저로 토큰화했을 때, 문서별 토큰 개수의 분포 그래프
- V100 GPU에서 학습 가능한 모델의 최대 토큰 길이는 일반적으로 512개로 제한
    
    → 이로 인해, 위키피디아 문서 중 20% 이상에서는 overflow token이 발생
    
- 문서의 중요한 내용이 후반부에 있을 경우, 모델이 해당 내용을 정확히 임베딩하지 못하는 문제 발생 가능
    
    → 이러한 문제를 해결하기 위해 overflow token을 학습에 포함하는 실험을 진행할 예정이다.
    
<br />

**4.2 질문 유형 분석**

<div display="flex">
    <img width="320" src="https://github.com/user-attachments/assets/e57248e7-2f34-45ff-9286-a777ecbf1ea8" />
    <img width="320" src="https://github.com/user-attachments/assets/5e0f7d64-33d7-488f-ae91-ad61f71f6b77" />
    <img width="320" src="https://github.com/user-attachments/assets/e791f61f-9ff4-4f2c-8095-6a504d6fb417" />
</div>

- 인물에 대한 질문이 가장 많음
- 인물의 이름과 같은 고유 명사는 토크나이징이 적절히 되지 않아서 이를 위한 처리가 필요할 것으로 생각

<br />

**4.3 train/validation 데이터셋의 context 수와 위키피디아 문서 수 비교**

<img width="450" src="https://github.com/user-attachments/assets/d4afcd18-d7db-42d3-9832-fadc865575c6" />


- Train 데이터셋에서는 일부 문서가 두 번 혹은 세 번 사용된 경우도 있으나, 전반적으로 매우 적은 빈도로 사용됨
- Validation 데이터셋의 경우는 거의 모든 문서가 한 번만 사용된 것으로 보임
- 대부분의 문서가 QA 데이터셋에서 한 번만 사용된 것으로 나타남
- 특정 문서가 QA 데이터에서 과도하게 사용되는 경우가 거의 없고, 다양한 문서가 QA 데이터셋에 고루 분포됨
    
    → train/validation 문서를 이용해 QA 데이터셋 증강 가능

<br />
    
<img width="450" src="https://github.com/user-attachments/assets/26268787-ba4d-4b0e-b0e8-54d49d9dc79d" />


- 5.8%의 문서만이 QA 데이터셋에서 사용되었고, 94.2%의 위키피디아 문서가 QA 데이터에서 사용되지 않음
- 아직 많은 문서가 QA 데이터셋에 포함되지 않았음을 의미
    
    → QA 데이터셋에 포함되지 않은 위키피디아 문서들을 대상으로 새로운 QA 질문을 생성하거나 데이터 증강을 시도할 수 있는 잠재적인 기회가 많이 존재한다고 볼 수 있음

<br />

**4.4 위키피디아 문서 분석**

- 위키피디아 문서 중 아래 예시와 같이 마크다운 태그, 주석 문구, URL, 인용 표시 등 문서 본문의 내용과 크게 관련 없는 문자열이 섞여 있는 경우가 일부 있음

  <img width="450" src="https://github.com/user-attachments/assets/c0ae9ce6-a55c-439f-996d-540ba59b4880" />

- 특히 TF-IDF retriever로 테스트해본 결과, `\`, `*` `p=` 등과 같이 마크다운, 인용 표시 등으로부터 추출된 토큰이 일부 문서에서 높은 점수를 얻는 경우도 확인됨
    
     → 문서의 내용과 관련 없는 불필요한 문자열을 HTML태그, 코드, 인용 표시, URL, 주석 문구 등으로 분류하고, 각 종류별로 적절하게 전처리 또는 제거해주는 실험 계획

<br />

## 5 프로젝트 수행

**5.1 Data Processing**

- Preprocessing: 원래 문서 + 제목 병합 후 전처리, 질문 전처리, 질문 카테고리 전처리, 마크다운 전처리, 위키피디아 문서 전처리
- Augmentation: 기존 질문 수정 및 증강, 기존 활용 문서 범위 새 질문 답변 생성 증강, 기존 미활용 문서 범위 새 질문 답변 생성 증강, ETRI_MRC 데이터셋 활용 증강
- Postprocessing: 조사 제거

**5.2 Retriever**

- Sparse Passage Retrieval (SPR): TF-IDF, BM25-subword, BM25-morphs, BM25-combine
- Dense Passage Retrieval (DPR) : overflow token, hard negative sampling, LoRA
- Re-ranking: Two-stage re-ranking (SPR → DPR)

**5.3 Reader**

- Transfer Learning
- Separate Inference
- LoRA

**5.4 Ensemble**

- Hard Voting

<br />

## 6 Wrap-up Report

자세한 내용은 <a href="https://github.com/boostcampaitech7/level2-mrc-nlp-11/blob/docs-55/upload_readme_report/ODQA%EB%8C%80%ED%9A%8C_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(11%EC%A1%B0).pdf">**Wrap-up Report**</a>를 참고해 주세요 !
