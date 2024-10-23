import wandb
import os

# WANDB 세션 시작
wandb.init(project="sangyeop", entity="7-11")

# 체크포인트 파일들이 있는 디렉토리 경로
# file_paths = [
#     "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/checkpoints/fine_tuned/uomnf97_4dataset-finetuned_happy-universe-77_original_default_bz=16_lr=1.7110631470130408e-05_fine_tuned_epoch=00_exact_match=72.50.ckpt",
#     "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/checkpoints/fine_tuned/klue-roberta-large_korquad1.0_filtered_classic-valley-65_original_default_bz=16_lr=1.6764783497920226e-05_fine_tuned_epoch=03_exact_match=71.67.ckpt"
# ]
directory_path = (
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/checkpoints/pre-trained"
)

# 새로운 Artifact 생성
# artifact = wandb.Artifact('2nd_fine_tuning_checkpoint', type='model')
artifact = wandb.use_artifact("2nd_fine_tuning_checkpoint:latest", type="model")
new_artifact = wandb.Artifact("2nd_fine_tuning_checkpoint", type="model")
# 기존 아티팩트의 파일들을 유지 (기존 파일 복사)
for file_name in artifact.manifest.entries:
    reference = f"artifact:{artifact.name}:{artifact.version}/{file_name}"
    new_artifact.add_reference(reference, name=file_name)

# 디렉토리 전체를 추가
# for file_path in file_paths:
#     artifact.add_file(file_path)
new_artifact.add_dir(directory_path)
# Artifact 저장
wandb.log_artifact(new_artifact)

# 세션 종료
wandb.finish()
