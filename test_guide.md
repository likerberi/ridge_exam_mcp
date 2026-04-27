# MCP 서버 테스트 가이드

이 가이드는 소규모 회계법인의 이상치 탐지 시나리오를 기준으로 MCP 서버를 검증하는 흐름입니다.

## 테스트 시나리오

목표는 비정형 재무 CSV를 수작업으로 전수 검토하던 흐름을, Ridge 기반 우선순위 검토 흐름으로 바꾸는 것입니다.

## 도구별 예시

### 1. 회계형 샘플 데이터 생성

```python
create_sample_data("test_data.csv", n_samples=200)
```

### 2. 데이터 로드 및 기본 정보 확인

```python
load_data("mcp_server/test_data.csv")
```

### 3. expected_audit_score 기준 전처리

```python
preprocess_data("mcp_server/test_data.csv", target_column="expected_audit_score")
```

### 4. Ridge 단독 분석

```python
ridge_analysis(
	"mcp_server/test_data_processed.csv",
	target_column="expected_audit_score",
	alpha=3.0,
)
```

### 5. OLS vs Ridge 비교와 이상치 우선순위화

```python
compare_ols_vs_ridge(
	"mcp_server/test_data_processed.csv",
	target_column="expected_audit_score",
	alpha=3.0,
	top_n=8,
)
```

### 6. 결과 시각화

```python
visualize_ridge_results(
	"mcp_server/test_data_processed.csv",
	target_column="expected_audit_score",
	alpha=3.0,
)
```

## Claude Desktop 프롬프트 예시

1. 샘플 회계 데이터를 생성해줘
2. 생성한 CSV를 읽고 컬럼, 결측치, 샘플 행을 요약해줘
3. expected_audit_score를 타겟으로 전처리해줘
4. OLS와 Ridge를 비교해서 왜 Ridge가 더 적합한지 설명해줘
5. 상위 8개 이상치 후보만 우선 검토할 수 있게 표로 정리해줘
6. 예측 결과와 잔차 플롯도 보여줘

## 검증 포인트

- 다중공선성 요약값이 반환되는지 확인
- OLS 대비 Ridge 성능 차이가 수치로 보이는지 확인
- top_anomalies에 검토 우선순위가 높은 행이 정렬되어 있는지 확인
- review_workflow에서 before/after 검토 건수 축소가 계산되는지 확인

## 실제 CSV 사용 시 주의점

- 타겟 컬럼은 수치형이어야 합니다.
- 전처리 단계에서 결측치가 제거되므로, 제거 행 수를 함께 확인하는 것이 좋습니다.
- 회계사가 검토할 수 있도록 top_n을 5~15 사이로 두면 운영하기 쉽습니다.
