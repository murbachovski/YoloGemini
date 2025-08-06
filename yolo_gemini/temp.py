import streamlit as st
import cv2
import time
import numpy as np
import google.generativeai as genai
from ultralytics import YOLO
from openapi_its import yolo_its
import ssl

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# Gemini API 설정
genai.configure(api_key="AIzaSyCog1z-WEO9pnM_zDXVqM7aFtIyIGpnNZk")
gen_model = genai.GenerativeModel("gemini-2.0-flash")

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🚦 실시간 객체 탐지 + Gemini 분석")

# 좌/우 레이아웃
col1, col2 = st.columns([2, 3])
video_placeholder = col1.empty()
image_placeholder = col2.empty()
text_placeholder = col2.empty()
analyze_btn = col2.button("📊 Gemini로 분석하기")

# CCTV 열기
cctv = yolo_its.get_its()
cap = cv2.VideoCapture(cctv)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 648)

# 마지막 탐지 결과 저장
last_frame = None
last_results = None

# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("카메라를 열 수 없습니다.")
        break

    # YOLO 탐지
    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # 좌측에 실시간 영상 표시
    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

    # 최근 프레임 & 결과 저장
    last_frame = rgb_frame
    last_results = results

    # 분석 버튼 눌렀을 때 Gemini 실행
    if analyze_btn and last_results:
        try:
            detected_classes = [int(cls) for cls in last_results[0].boxes.cls]
            class_names = [model.names[i] for i in detected_classes]
            prompt = f"다음 객체들이 탐지되었습니다: {', '.join(class_names)}. 이 장면에 대해 설명해주세요."

            with st.spinner("Gemini가 분석 중입니다..."):
                answer = gen_model.generate_content(prompt).text

            image_placeholder.image(last_frame, channels="RGB", caption="분석된 장면")
            text_placeholder.subheader("🧠 Gemini 분석 결과")
            text_placeholder.write(answer)

        except Exception as e:
            text_placeholder.error(f"Gemini 오류: {e}")

    # 프레임 속도 조절 (30fps 기준 약 0.03초)
    time.sleep(0.03)

cap.release()
