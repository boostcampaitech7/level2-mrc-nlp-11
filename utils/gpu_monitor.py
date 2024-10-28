#!/usr/bin/env python3

import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------- 설정 부분 ----------------------
# SENDER_PASSWORD는 아래 링크를 참고
# https://kincoding.com/entry/Google-Gmail-SMTP-%EC%82%AC%EC%9A%A9%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%84%B8%ED%8C%85
# 이메일 설정
SMTP_SERVER = "smtp.gmail.com"  # SMTP 서버 주소 (예: Gmail)
SMTP_PORT = 587  # SMTP 포트 (Gmail: 587)
SENDER_EMAIL = "google@gmail.com"  # 예시 발송자 이메일 주소
SENDER_PASSWORD = "goog leap pleo pena"  # 예시 발송자 이메일 비밀번호 또는 앱 비밀번호
RECIPIENT_EMAIL = "apple@icloud.com"  # 수신자 이메일 주소

# 모니터링 설정
CHECK_INTERVAL = 30  # 초 단위로 GPU 상태 확인 (예: 60초)

# --------------------------------------------------------


def send_email(subject, body):
    """이메일 발송 함수"""
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        print(f"[INFO] 이메일 발송 성공: {subject}")
    except Exception as e:
        print(f"[ERROR] 이메일 발송 실패: {e}")


def is_gpu_idle():
    """GPU가 사용 중인지 확인하는 함수"""
    try:
        # nvidia-smi 명령어 실행
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"[ERROR] nvidia-smi 실행 실패: {result.stderr}")
            return False  # 오류 발생 시 GPU가 사용 중이라고 가정

        # PID 목록을 가져옴
        pids = result.stdout.strip().split("\n")
        # 비어있으면 GPU가 사용되지 않음
        if pids == [""]:
            return True
        else:
            return False
    except Exception as e:
        print(f"[ERROR] GPU 상태 확인 중 오류 발생: {e}")
        return False  # 오류 발생 시 GPU가 사용 중이라고 가정


def main():
    """메인 함수"""
    alert_sent = False  # 알림이 발송되었는지 여부

    while True:
        idle = is_gpu_idle()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if idle:
            print(f"[{current_time}] GPU is idle.")
            if not alert_sent:
                subject = "GPU Idle Alert"
                body = (
                    f"As of {current_time}, the GPU is idle. You can submit your job."
                )
                send_email(subject, body)
                alert_sent = True  # 알림 발송 상태 업데이트
        else:
            print(f"[{current_time}] GPU is in use.")
            alert_sent = False  # GPU가 사용 중일 때 알림 상태 초기화

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
