# mecro
파이썬 블루스택 메크로
## 1. 개요

### 1.1 목적

본 가이드는 **Python**으로 **BlueStacks** 기반 모바일 게임을 자동화하는 **CodeX Agent**를 설계·구현·배포·유지보수하기 위한 종합 안내서다. 초급 개발자가 최소한의 학습 곡선으로 프로젝트에 기여할 수 있도록 모든 예제와 주석은 **한국어**로 작성되며, 모듈 구조를 명확히 분리해 향후 기능 추가 및 유지보수가 쉽도록 돕는다.

### 1.2 대상 독자

- Python에 막 입문했지만 실제 프로젝트로 실력을 키우고 싶은 개발자
- BlueStacks/ADB 자동화 흐름을 배우고자 하는 QA·RPA·매크로 개발자
- 기능 추가·버그 픽스를 맡게 될 유지보수 담당자
- 프로젝트 매니저·리더

## 2. CodeX Agent 정의

### 2.1 역할

- BlueStacks 에뮬레이터 화면을 캡처하여 템플릿 매칭으로 UI 요소를 탐색
- ADB 명령으로 터치·드래그·키 입력 등 게임 조작 수행
- 사용자 정의 **시퀀스(Sequence)** 로직으로 자동 사냥·반복 작업 등 수행
- 로그/스케줄 관리, 예외 처리, 결과 리포트

### 2.2 주요 기능

1. **스크린샷 캡처** : 고속 이미지 캡처 → numpy ndarray 변환
2. **템플릿 매칭** : OpenCV TM\_CCOEFF\_NORMED 기반; ROI 지정 가능
3. **입력 이벤트** : tap, swipe, keyevent, drag & drop 지원
4. **시퀀스 엔진** : JSON/YAML 파일에 선언형으로 정의
5. **로깅** : 표준 logging 모듈, 회전 로그 핸들러
6. **설정 관리** : `.env` + `config.yaml` 두 단계
7. **확장 플러그인** : 새 게임·새 화면에 맞춘 모듈 핫스왑

### 2.3 구성 요소

| 모듈                    | 파일                    | 책임                      |
| --------------------- | --------------------- | ----------------------- |
| **adb\_controller**   | `adb_controller.py`   | ADB 연결, 명령 송신, 터치/키 입력  |
| **screen\_capture**   | `screen_capture.py`   | BlueStacks 창 찾기, 캡처, 변환 |
| **template\_matcher** | `template_matcher.py` | OpenCV 기반 매칭 및 점수 계산    |
| **sequence\_manager** | `sequence_manager.py` | 시퀀스 DSL 파싱, 실행, 예외 처리   |
| **config**            | `config.yaml`         | 경로·ROI·임계값·시퀀스 정의       |
| **main**              | `main.py`             | CLI 진입점, 인수 파싱          |

## 3. 설치 및 설정

### 3.1 시스템 요구사항

- Windows 10/11 (x64)
- Python 3.10 이상 (pip)
- BlueStacks 5.20 이상
- ADB (Android SDK Platform‑Tools 34+)
- 메모리 8 GB 이상, SSD 권장

### 3.2 설치 절차

```shell
# 1. 레포 클론
git clone https://github.com/your-org/codex-agent.git
cd codex-agent

# 2. 가상 환경
python -m venv venv
venv\Scripts\activate

# 3. 필수 패키지
pip install -r requirements.txt  # 필수 패키지 목록은 3.4절 참고

# 4. ADB 환경 변수
setx ADB_PATH "C:\Android\platform-tools\adb.exe"
```

### 3.3 환경 변수 및 구성파일

`config.yaml` 예시:

```yaml
adb_path: "${ADB_PATH}"
bluestacks_title: "BlueStacks App Player"
roi_default: [0, 0, 1971, 1131]  # X0, Y0, X1, Y1
match_threshold: 0.8
sequences_file: "sequences.yaml"
```

### 3.4 필수 Python 패키지 (requirements.txt)

필수 패키지를 관리하기 위해 `requirements.txt` 파일을 사용합니다. 기본 구성은 다음과 같습니다.

```text
opencv-python
numpy
psutil
ppadb-py3
python-dateutil
```

> 필요에 따라 `pytest`, `pytest-mock`, `black`, `isort`, `flake8` 등을 추가하세요.

## 4. 기본 사용법

### 4.1 시작하기

```shell
python main.py --seq "펫 성장"
```

첫 실행 시 템플릿 PNG 파일이 `assets/templates`에 위치하는지 확인.

### 4.2 주요 CLI 인수

| 옵션           | 설명                     |
| ------------ | ---------------------- |
| `--seq NAME` | 수행할 시퀀스 이름             |
| `--roi N`    | ROI y0 기준값(픽셀). 기본 600 |
| `--once`     | 한 번만 실행 후 종료           |
| `--save`     | 매칭 결과 사각형 디버그 이미지 저장   |

### 4.3 워크플로 예시

1. `sequences.yaml` 에 시퀀스 선언
2. 템플릿 PNG 저장 → 이름 동일하게 맵핑
3. `python main.py --seq "쫄작완료"` 실행
4. 로그/스크린샷 확인

## 5. 고급 주제

### 5.1 플러그인/모듈 확장

- `templates/MyGame/*.png` 폴더 구조 그대로 추가
- `sequences.yaml` 에 새 시퀀스 블록 삽입
- 필요 시 `plugins/` 안에 전용 모듈 작성 후 `main.py`에서 동적 import

### 5.2 자동화 스크립트 작성 패턴

```python
# 예) 아이템 가방 열기 함수 (한국어 주석)
def open_inventory(bs) -> bool:
    """
    가방 아이콘을 찾아 탭하여 인벤토리 창을 연다.
    :param bs: BlueStacks 컨트롤 객체
    :return: 성공 여부
    """
    icon = tpl_match("가방.png", threshold=0.9)
    if icon:
        adb_tap(icon.center)
        return True
    return False
```

### 5.3 트러블슈팅

| 증상                                                | 원인              | 해결 방법                          |
| ------------------------------------------------- | --------------- | ------------------------------ |
| OpenCV `-215:Assertion failed`                    | 템플릿 크기가 ROI보다 큼 | 템플릿 해상도 축소 또는 ROI 확장           |
| `ValueError: not enough values to unpack`         | 함수 반환값 개수 불일치   | 시퀀스 정의와 함수 시그니처 맞추기            |
| `BlueStacks object has no attribute 'screenshot'` | 버전 호환 안 됨       | `screen_capture.py` 의 함수 참조 수정 |

## 6. 보안 및 컴플라이언스

- 사내 정책상 게임 자동화 툴 배포 시 실행 스크립트에 라이선스 고지 포함
- 로그에 개인정보(계정명, 토큰) 저장 금지 → `.gitignore`에 `logs/*` 포함

## 7. FAQ

1. **Q: 템플릿 매칭이 자주 실패합니다.**\
   **A:** 해상도 맞춤·ROI 조정·`match_threshold` 값을 0.75로 낮춰보세요.

2. **Q: 여러 BlueStacks 인스턴스를 제어할 수 있나요?**\
   **A:** `adb -s emulator-5555` 처럼 디바이스 ID를 인수로 전달하면 가능합니다.

## 8. 코드 작성 가이드라인 (한국어 주석)

1. **주석/문서화**

   - 모든 함수·클래스 docstring은 **한국어**.
   - 간단한 한글 설명 + 타입 힌트 명시.
   - 변경 이력은 `CHANGELOG.md` 로 관리.

2. **코딩 스타일**

   - PEP 8 기본 + 120자 라인 길이
   - `black`, `isort`, `flake8` 프리훅 설정
   - 함수 이름은 snake\_case, 클래스는 PascalCase

3. **모듈 구조**

   - 기능별 디렉터리 (`core/`, `plugins/`, `assets/`)
   - 복잡한 로직은 클래스화, 싱글턴은 피함

4. **설정 분리**

   - 하드코딩 금지, 모든 상수는 `config.yaml` 또는 `.env`
   - 테스트용 설정 파일 별도(`config_test.yaml`)

5. **예외 처리**

   - 모든 외부 자원 호출(ADB, 파일 IO) try‑except
   - 사용자 친화 에러 메시지(한국어) 로깅

6. **유닛 테스트**

   - `pytest` + `pytest-mock`, 커버리지 80% 목표
   - 이미지 비교는 `assert abs(val - expected) < 0.01`

## 9. 참고 자료

- ADB 공식 문서: [https://developer.android.com/tools/adb](https://developer.android.com/tools/adb)
- OpenCV TM 매칭 가이드
- Python PEP 8 스타일 가이드
- 블로그: “BlueStacks 자동화 실전 팁 모음”

