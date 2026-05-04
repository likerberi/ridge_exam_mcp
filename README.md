# Ridge MCP Server

Python MCP server for ridge-based accounting anomaly analysis in Claude Desktop.

소규모 회계법인에서 비정형 재무 데이터를 수작업으로 훑으며 이상치를 찾던 흐름을, Claude Desktop 안에서 자연어로 실행 가능한 분석 워크플로로 바꾼 Ridge 기반 MCP 서버입니다.

## 문제 정의

현업 회계사는 거래처별 매출, 미수금, 급여, 비용청구, 출장비 같은 재무 컬럼이 서로 강하게 엮인 CSV를 엑셀로 열어 보고, 행 단위로 수상한 항목을 직접 추려야 하는 경우가 많습니다. 이 방식은 시간이 오래 걸리고, 어떤 행을 먼저 봐야 하는지 우선순위를 잡기 어렵다는 문제가 있습니다.

이 프로젝트는 그 과정을 두 단계로 바꿉니다.

1. 회계 데이터의 다중공선성을 고려해 Ridge 회귀로 기대값을 추정합니다.
2. 실제값과 예측값의 잔차를 기준으로 검토 우선순위를 자동으로 정렬합니다.

## 왜 Ridge인가

회계 데이터에서는 매출과 미수금, 급여와 투입시간, 비용청구와 출장비처럼 설명변수끼리 강하게 상관되는 경우가 흔합니다. 이런 상황에서 OLS는 계수가 불안정해지고 샘플이 조금만 바뀌어도 예측이 흔들릴 수 있습니다.

Ridge는 $L_2$ regularization으로 계수 폭주를 줄여 다중공선성이 심한 재무 데이터에서도 더 안정적인 예측 기준선을 제공합니다. 이 저장소에는 OLS와 Ridge를 직접 비교하는 compare_ols_vs_ridge 도구가 포함되어 있어, 왜 regularization이 필요한지 숫자로 확인할 수 있습니다.

## 왜 MCP인가

대상 사용자는 개발자가 아니라 회계사입니다. 분석 로직이 Python 코드에만 있으면, 실제 업무에서 쓰기 어렵습니다.

MCP로 서버를 감싸면 Claude Desktop에서 아래처럼 자연어로 바로 실행할 수 있습니다.

- 샘플 회계 데이터를 만들어줘
- 데이터 전처리 후 Ridge와 OLS를 비교해줘
- 이상치 후보 상위 8건만 우선 검토하게 정리해줘

즉, 모델링 코드를 API나 노트북으로 직접 다루지 않아도 회계사가 분석을 실행할 수 있게 만드는 것이 MCP를 쓴 이유입니다.

## 결과와 임팩트

저장소의 회계형 샘플 데이터를 실제로 생성하고, 전처리 후 compare_ols_vs_ridge를 실행한 결과는 아래와 같습니다.

- 데이터 규모: 원본 180행, 전처리 후 171행
- OLS 대비 Ridge 테스트 MSE 2.12% 감소
- OLS 대비 Ridge 테스트 $R^2$ 0.0029 개선
- 계수 $L_2$ norm 8.71% 감소
- 수작업 전체 검토 대상 35건에서 상위 8건 우선 검토로 축소
- 검토량 77.14% 감소

첫 번째 이상치 후보는 실제값 36,508 대비 예측값 43,600으로 절대 잔차 7,091, 잔차 z-score 3.27을 기록했습니다. 즉, 이전에는 테스트 구간 전체 35건을 사람이 훑어야 했다면, 이제는 상위 8건부터 집중해서 보는 식으로 업무 순서를 바꿀 수 있습니다.

## 주요 기능

- CSV 로드 및 결측치/스키마 점검
- 수치형 컬럼 표준화와 전처리 파일 생성
- Ridge 회귀 분석 및 계수 확인
- OLS 대비 Ridge 성능 비교
- 잔차 기반 이상치 후보 우선순위화
- 예측 결과 및 잔차 시각화
- Claude Desktop 자연어 실행 지원

## 사전 요구사항

- Python 3.10 이상
- uv 패키지 관리자
- Claude Desktop

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/likerberi/ridge_exam_mcp
cd ridge_exam_mcp
```

### 2. 의존성 설치

```bash
uv sync
```

### 3. Claude Desktop 설정

Claude Desktop 설정 파일에 아래 서버를 추가합니다.

- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
- Windows: %APPDATA%\Claude\claude_desktop_config.json

```json
{
  "mcpServers": {
    "ridge-analysis": {
      "command": "uv",
      "args": ["run", "python", "mcp_server/server.py"],
      "cwd": "<YOUR_PROJECT_PATH>/ridge_exam_mcp"
    }
  }
}
```

`claude_desktop_config.json` 예시 파일도 동일한 플레이스홀더를 사용하므로, 실제 사용 시 본인 환경의 절대 경로로 바꿔야 합니다.

설정 후 Claude Desktop을 재시작합니다.

## Claude Desktop에서 쓰는 방법

### 인풋 → 아웃풋 가이드

각 단계에서 무엇을 준비하고 무엇이 돌아오는지 정리합니다.

---

#### 1단계. 샘플 데이터 생성 (처음 시작하는 경우)

| | 내용 |
|---|---|
| **인풋** | 없음 (또는 파일명, 생성할 행 수) |
| **요청 예시** | `샘플 회계 데이터를 200행으로 만들어줘` |
| **아웃풋** | 생성된 CSV 파일 경로, 컬럼 목록, 삽입된 이상치 행 번호, 결측치 현황 |

> 실제 데이터가 있으면 이 단계를 건너뛰고 2단계부터 시작합니다.

---

#### 2단계. 데이터 로드 및 구조 확인

| | 내용 |
|---|---|
| **인풋** | CSV 파일 경로 |
| **요청 예시** | `mcp_server/test_data.csv 파일을 읽고 컬럼과 결측치 현황을 알려줘` |
| **아웃풋** | 전체 행/열 수, 컬럼명 목록, 컬럼별 데이터 타입, 결측치 개수, 상위 5행 미리보기 |

---

#### 3단계. Benford's Law 검증 (데이터 조작 여부 1차 점검)

| | 내용 |
|---|---|
| **인풋** | CSV 파일 경로, (선택) 점검할 컬럼명 |
| **요청 예시** | `expense_claims와 vendor_spend에 Benford 검증을 해줘` |
| **아웃풋** | 컬럼별 `PASS / WARNING / CRITICAL` 플래그, 카이제곱 통계량, p-value, 끝자리 0 집중 비율(round number bias) |

> `CRITICAL` (p < 0.01): 데이터 조작 또는 비정상 입력 강하게 의심
> `WARNING` (p < 0.05): 추가 확인 권장
> `PASS`: 정상 분포 범위

---

#### 4단계. 전처리

| | 내용 |
|---|---|
| **인풋** | CSV 파일 경로, 예측 기준이 될 타겟 컬럼명 |
| **요청 예시** | `expected_audit_score를 타겟으로 전처리해줘` |
| **아웃풋** | 전처리된 CSV 파일 경로(`_processed.csv`), 제거된 결측치 행 수, 표준화 적용 컬럼 목록 |

---

#### 5단계. OLS vs Ridge 비교 및 이상치 우선순위화

| | 내용 |
|---|---|
| **인풋** | 전처리된 CSV 파일 경로, 타겟 컬럼명, (선택) 상위 몇 건 볼지 |
| **요청 예시** | `이상치 후보 상위 8건을 우선순위 순으로 정리해줘` |
| **아웃풋** | 자동 선택된 최적 alpha 값, OLS/Ridge 각각의 MSE·R², 계수 L2 norm 비교, 잔차 기준 이상치 후보 테이블(행 번호·실제값·예측값·잔차·z-score), 검토량 절감률 |

> alpha는 자동으로 교차검증(RidgeCV)으로 선택됩니다. 직접 지정도 가능합니다.

---

#### 6단계. 시각화

| | 내용 |
|---|---|
| **인풋** | 전처리된 CSV 파일 경로, 타겟 컬럼명 |
| **요청 예시** | `예측 결과와 잔차 플롯을 보여줘` |
| **아웃풋** | 실제값 vs 예측값 산점도, 잔차 플롯 (이미지) |

---

### 전체 흐름 요약

```
실제 CSV 보유 시:  로드 → Benford 검증 → 전처리 → 비교 분석 → 시각화
처음 시작하는 경우: 샘플 생성 → 로드 → 전처리 → 비교 분석 → 시각화
```

## 제공 도구

현재 MCP 서버는 아래 7개 도구를 제공합니다.

| 도구명 | 설명 | 주요 매개변수 |
|--------|------|---------------|
| create_sample_data | 다중공선성이 있는 회계형 샘플 데이터 생성 | filename, n_samples |
| load_data | CSV 파일 로드 및 기본 정보 확인 | filepath |
| benford_analysis | Benford's Law 기반 데이터 조작 여부 검증 | filepath, columns |
| preprocess_data | 결측치 제거 및 수치 데이터 표준화 | filepath, target_column |
| ridge_analysis | Ridge 회귀 모델 훈련 및 평가 (alpha 자동 선택) | filepath, target_column, test_size, alpha |
| compare_ols_vs_ridge | OLS 대비 Ridge 성능 비교와 이상치 우선순위화 (alpha 자동 선택) | filepath, target_column, test_size, alpha, top_n |
| visualize_ridge_results | 실제값 vs 예측값, 잔차 플롯 생성 | filepath, target_column, alpha |

## 직접 실행

```bash
uv run python mcp_server/server.py
```

## 프로젝트 구조

```text
ridge_exam_mcp/
├── mcp_server/
│   └── server.py
├── pyproject.toml
├── claude_desktop_config.json
├── test_guide.md
└── README.md
```

## 참고 자료

- https://modelcontextprotocol.io/
- https://github.com/modelcontextprotocol/python-sdk
- https://claude.ai/download
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

## 라이선스

MIT License
