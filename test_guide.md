# MCP 서버 테스트 스크립트

이 스크립트는 릿지 분석 MCP 서버의 기능을 테스트합니다.

## 사용 가능한 도구:

### 1. create_sample_data
```python
# 샘플 데이터 생성
create_sample_data("test_data.csv", n_samples=200)
```

### 2. load_data 
```python
# 데이터 로드 및 기본 정보 확인
load_data("test_data.csv")
```

### 3. preprocess_data
```python 
# 데이터 전처리 (결측치 제거 및 표준화)
preprocess_data("test_data.csv", target_column="target")
```

### 4. ridge_analysis
```python
# 릿지 회귀 분석 수행
ridge_analysis("test_data_processed.csv", target_column="target", alpha=1.0)
```

### 5. visualize_ridge_results
```python
# 결과 시각화
visualize_ridge_results("test_data_processed.csv", target_column="target", alpha=1.0)
```

## Claude Desktop에서 사용법:

1. "샘플 데이터를 생성해주세요"
2. "데이터를 로드하고 기본 정보를 알려주세요"  
3. "데이터를 전처리해주세요"
4. "릿지 회귀 분석을 실행해주세요"
5. "결과를 시각화해주세요"

## 실제 CSV 파일 사용:
- 본인의 CSV 파일 경로를 지정하여 분석 가능
- 타겟 컬럼명을 정확히 입력 필요
