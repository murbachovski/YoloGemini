# 프로젝트 제목
```
YoloGemini — ITS CCTV 영상 Gemini 분석 서비스
```

# 프로젝트 설명
```
OpenAPI ITS CCTV 영상을 가져와서,
Google Gemini AI로 영상을 분석해 자연어 답변을 제공하는 웹 서비스
```

# 가상환경 설정
```
conda create -n yolo_gem python=3.9
```

# API_KEY 설정
```
export GENAI_API_KEY=""
export ITS_API_KEY=""
```

# 라이브러리 설치
```
pip install -r requirements.txt
```

# 앱 실행
```
./run.sh
```

# 웹 구성
<p align="center">
  <img src="https://github.com/user-attachments/assets/bb237c58-53e5-4a7e-975a-ff504aa4684e" width="700">
  <img src="https://github.com/user-attachments/assets/a70c70d7-57d9-4907-8806-4e98b26b9caa" width="700">
  <img src="https://github.com/user-attachments/assets/93685bbe-1136-4e18-b20b-5a9952e7c237" width="700">
  <img src="https://github.com/user-attachments/assets/8b9bef0d-7d25-4eee-999c-7ec33fe6d586" width="700">
  <img src="https://github.com/user-attachments/assets/f0912d8f-ec16-4b41-a718-dcf7dfa48157" width="700">
  <img src="https://github.com/user-attachments/assets/b2df780a-36a8-4153-acbf-a52e737f303e" width="700">
</p>

# Ngrok
(로컬 서버 => 공개 서버로 전환)
```
<Mac M1 설치 기준>
https://ngrok.com/downloads/mac-os
brew install ngrok
ngrok config add-authtoken <token>
ngrok http 80
```

# Ngrok log
<p align="center">
  <img src="https://github.com/user-attachments/assets/5ca755c3-d8f8-4088-b3b4-1b735945d351" width="700">
</p>

# Ngrok(공개 서버 접속)
[Ngrok 공개 서버 접속](https://c83c0967a9dd.ngrok-free.app/)<br>

# Ngrok 참고 문서
[위키독스](https://cordcat.tistory.com/105)<br>

# Make requirements.txt
```
pip install pipreqs
```

# pipreqs 참고 문서
[PyPI pipreqs](https://pypi.org/project/pipreqs/)<br>

# YoloGemini 참고 문서
[YoloGemini](https://youtu.be/wVPQzJgnUC4?si=3H6xq1bqheE7S6bq)<br>

