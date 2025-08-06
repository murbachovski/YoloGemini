# OpenAPI
    # ITS
        # https://its.go.kr/opendata/
# 가상환경 생성
    # conda create -n oapi python=3.9
    # pip install pandas opencv-python ultralytics

# 📦 필요한 라이브러리 불러오기
import json
import pandas as pd
import urllib.request
import os
from typing import Tuple, Optional

ITS_API_KEY: Optional[str] = os.getenv("ITS_API_KEY")

def get_its():
    # 🔑 1. Open API 인증키
    api_key = ITS_API_KEY
    
    # 🛣️ 2. 도로 유형 선택 (its: 일반도로 / ex: 고속도로)
    road_type = "its"

    # 🌍 3. CCTV 요청 범위 설정 (서울 지역 기준)
    minX, maxX = 126.76, 127.18  # 경도 (서울)
    minY, maxY = 37.41, 37.70    # 위도 (서울)

    # 📂 4. 응답 포맷 설정
    response_type = "json"

    # 🌐 5. CCTV 정보 요청 URL 구성
    url = (
        f"https://openapi.its.go.kr:9443/cctvInfo?"
        f"apiKey={api_key}&type={road_type}&cctvType=1"
        f"&minX={minX}&maxX={maxX}&minY={minY}&maxY={maxY}"
        f"&getType={response_type}"
    )

    # 📡 6. API 요청 및 응답 수신
    response = urllib.request.urlopen(url)
    json_str = response.read().decode("utf-8")
    json_data = json.loads(json_str)

    # 📊 7. JSON 응답을 DataFrame으로 변환
    cctv_df = pd.json_normalize(json_data["response"]["data"], sep='')

    # 🔍 8. 테스트용 CCTV 스트리밍 URL 하나 선택
    test_url = cctv_df['cctvurl'][0]  # CCTV URL이 존재하는 경우 인덱스 조정 가능
    
    return test_url

