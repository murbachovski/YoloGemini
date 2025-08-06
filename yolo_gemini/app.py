"""
Streamlit을 이용한 실시간 CCTV 객체 탐지 및 Gemini 분석 애플리케이션 (v4)

이 애플리케이션은 다음 기능을 수행합니다:
1. CCTV 연결 시 프로그레스바를 통해 사용자에게 진행 상황을 표시합니다.
2. 모든 영상 프레임을 640x480 해상도로 통일하고, UI에 동일한 크기로 표시합니다.
3. (개선) 왼쪽의 실시간 영상과 오른쪽의 분석 결과 이미지의 세로 위치를 정렬합니다.
4. YOLOv8 모델을 사용하여 객체를 실시간으로 탐지합니다.
5. 'Gemini로 분석하기' 버튼 클릭 시, 버튼 아래에 스피너가 나타나 분석 중임을 알립니다.
6. 해당 시점의 '탐지 결과가 표시된' 이미지와 객체 목록을 UI에 표시하고,
   '원본' 이미지와 객체 목록은 Gemini API로 전송하여 분석을 요청합니다.
"""

import streamlit as st
import cv2
import time
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO
from openapi_its import yolo_its
import ssl
import os
from typing import Tuple, Optional

GENAI_API_KEY: Optional[str] = os.getenv("GENAI_API_KEY")

# --- 상수 정의 ---
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# --- 초기화 및 설정 함수들 ---

def initialize_app():
    """애플리케이션 실행에 필요한 초기 설정을 수행합니다."""
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        genai.configure(api_key=GENAI_API_KEY)
        gen_model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.error(f"Gemini API 키 설정 중 오류 발생: {e}")
        st.stop()
    try:
        yolo_model = YOLO('yolo11n.pt')
    except Exception as e:
        st.error(f"YOLO 모델을 로드하는 중 오류가 발생했습니다: {e}")
        st.stop()
    return gen_model, yolo_model

def initialize_session_state():
    """Streamlit의 세션 상태 변수들을 초기화합니다."""
    if 'analyze' not in st.session_state:
        st.session_state.analyze = False
        st.session_state.analysis_frame = None
        st.session_state.analysis_display_frame = None
        st.session_state.analysis_results = None
        st.session_state.gemini_answer = ""
        st.session_state.latest_frame = None
        st.session_state.latest_annotated_frame = None

# --- UI 설정 및 비즈니스 로직 ---

def trigger_analysis_callback():
    """'Gemini로 분석하기' 버튼 클릭 시 실행될 콜백 함수입니다."""
    if st.session_state.latest_frame is not None and st.session_state.latest_results is not None:
        st.session_state.analyze = True
        st.session_state.analysis_frame = st.session_state.latest_frame
        st.session_state.analysis_display_frame = st.session_state.latest_annotated_frame
        
        results = st.session_state.latest_results
        detected_classes = [int(cls) for cls in results[0].boxes.cls]
        yolo_model = st.session_state.yolo_model
        st.session_state.analysis_results = [yolo_model.names[i] for i in detected_classes]
        st.session_state.gemini_answer = ""

def perform_gemini_analysis(gen_model):
    """Gemini API를 호출하여 이미지 분석을 수행합니다."""
    try:
        frame_to_analyze = st.session_state.analysis_frame
        class_names = st.session_state.analysis_results
        pil_image = Image.fromarray(frame_to_analyze)

        prompt = f"""
        다음은 CCTV 화면에서 탐지된 객체 목록입니다: {', '.join(class_names) if class_names else '없음'}.
        이 정보를 바탕으로, 다음 질문에 답해주세요:
        1. 이 장면은 어떤 상황으로 보이나요? (예: 일반적인 도로 상황, 교통 정체, 보행자 이동 등)
        2. 특별히 주목할 만한 점이나 잠재적 위험 요소가 있나요?
        3. 각 객체들의 관계를 설명해주세요.
        4. 한 문장으로 요약 설명해주세요.
        """
        response = gen_model.generate_content([prompt, pil_image])
        st.session_state.gemini_answer = response.text
    except Exception as e:
        st.session_state.gemini_answer = f"Gemini 분석 중 오류 발생: {e}"
    finally:
        st.session_state.analyze = False

def display_analysis_results(image_placeholder, text_placeholder):
    """저장된 분석 결과를 UI에 표시합니다."""
    if st.session_state.analysis_display_frame is not None:
        caption_text = f"분석 시점의 화면 (탐지된 객체: {len(st.session_state.analysis_results)}개)"
        image_placeholder.image(
            st.session_state.analysis_display_frame,
            caption=caption_text,
            width=TARGET_WIDTH
        )
    if "오류" in st.session_state.gemini_answer:
        text_placeholder.error(st.session_state.gemini_answer)
    elif st.session_state.gemini_answer:
        text_placeholder.info(st.session_state.gemini_answer)

def run_video_loop(yolo_model, video_placeholder, progress_bar):
    """메인 비디오 처리 루프를 실행하고, 초기 연결 상태를 프로그레스바로 업데이트합니다."""
    try:
        cctv_url = yolo_its.get_its()
        cap = cv2.VideoCapture(cctv_url)
        progress_bar.progress(30, text="CCTV 스트림에 연결 중...")
        if not cap.isOpened():
            raise ConnectionError(f"CCTV 스트림을 열 수 없습니다: {cctv_url}")
        progress_bar.progress(60, text="스트림 정보 수신 중...")
    except Exception as e:
        st.error(f"CCTV 연결에 실패했습니다: {e}")
        progress_bar.empty()
        st.stop()

    is_first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("CCTV 스트림에서 프레임을 가져올 수 없습니다. 5초 후 재연결을 시도합니다.")
            time.sleep(5)
            continue
        
        if is_first_frame:
            progress_bar.progress(100, text="영상 출력 중!")
            # time.sleep(0.5)
            time.sleep(0.3)
            progress_bar.empty()
            is_first_frame = False

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        results = yolo_model(frame, conf=0.5, verbose=True)
        annotated_frame = results[0].plot()
        rgb_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(rgb_annotated_frame, width=TARGET_WIDTH)

        st.session_state.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.latest_annotated_frame = rgb_annotated_frame
        st.session_state.latest_results = results

        time.sleep(0.01)

    cap.release()
    st.info("CCTV 연결이 종료되었습니다.")

# --- 메인 실행 함수 ---

def main():
    """애플리케이션의 메인 실행 흐름을 제어합니다."""
    st.set_page_config(page_title="실시간 객체 탐지 및 분석", layout="wide")
    
    gen_model, yolo_model = initialize_app()
    initialize_session_state()
    st.session_state.yolo_model = yolo_model

    st.title("🚦 실시간 객체 탐지 + Gemini 분석")
    st.info("왼쪽에는 실시간 CCTV 영상이, 오른쪽에는 'Gemini로 분석하기' 버튼이 있습니다. 버튼을 누르면 해당 시점의 화면과 분석 결과가 오른쪽에 나타납니다.")

    # col1, col2 = st.columns([2, 3])
    col1, col2 = st.columns([1, 1])  # 동일한 너비로 조정
    with col1:
        st.subheader("📹 실시간 탐지 영상")
        video_placeholder = st.empty()
        progress_bar = st.progress(0, text="CCTV 연결 준비 중...")

    with col2:
        st.subheader("📊 분석 결과")
        
        # (수정) 플레이스홀더를 먼저 정의하여 세로 정렬을 맞춥니다.
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        
        # (수정) 버튼을 이미지와 텍스트 아래에 위치시킵니다.
        st.button("📊 Gemini로 분석하기", key="gemini_analysis_button", on_click=trigger_analysis_callback)

        if st.session_state.analyze:
            # 분석 로직은 플레이스홀더 아래에서 실행되도록 합니다.
            # 스피너는 텍스트 플레이스홀더를 임시로 사용하여 표시할 수 있습니다.
            text_placeholder.info("🧠 Gemini가 장면을 분석 중입니다...")
            perform_gemini_analysis(gen_model)
            display_analysis_results(image_placeholder, text_placeholder)
        else:
            # 분석 요청이 없을 때 이전에 분석된 결과가 있다면 계속 표시
            display_analysis_results(image_placeholder, text_placeholder)
            
    # 모든 UI가 구성된 후 비디오 루프 실행
    run_video_loop(yolo_model, video_placeholder, progress_bar)

if __name__ == "__main__":
    main()