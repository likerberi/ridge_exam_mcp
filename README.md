# Ridge MCP Server

릿지 회귀 분석을 위한 Model Context Protocol (MCP) 서버입니다.
Claude Desktop과 연동하여 데이터 분석 작업을 수행할 수 있습니다.

## 🚀 주요 기능
- **데이터 로드**: CSV 파일 로드 및 기본 정보 확인
- **데이터 전처리**: 결측치 제거 및 수치 데이터 표준화  
- **릿지 분석**: 릿지 회귀 모델 훈련 및 평가
- **시각화**: 예측 결과 및 잔차 플롯 생성
- **샘플 데이터**: 테스트용 샘플 데이터 자동 생성
- **Claude Desktop 연동**: MCP를 통한 자연어 인터페이스

## 📋 사전 요구사항
- Python 3.10 이상
- uv 패키지 관리자
- Claude Desktop 애플리케이션

## 🛠 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd ridge
```

### 2. 패키지 설치
```bash
uv add mcp[cli] pandas scikit-learn matplotlib numpy
```

### 3. Claude Desktop 설정
Claude Desktop 설정 파일에 MCP 서버를 추가하세요:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ridge-analysis": {
      "command": "uv", 
      "args": ["run", "python", "mcp_server/server.py"],
      "cwd": "c:/Users/우리집/project/ridge"
    }
  }
}
```

### 4. Claude Desktop 재시작
설정 파일 수정 후 Claude Desktop을 재시작하세요.

## 🎯 사용법
클로드 데스크탑과의 연동을 확인한 후 사용하세요.
(망치 모양이 뜨지만, 아직 베타인 관계로 안뜨는 버그가 있다고 함)

### Claude Desktop에서 사용
1. **샘플 데이터 생성**: "샘플 데이터를 생성해주세요"
2. **데이터 로드**: "데이터를 로드하고 정보를 보여주세요"  
3. **데이터 전처리**: "데이터를 전처리해주세요"
4. **릿지 분석**: "릿지 회귀 분석을 실행해주세요"
5. **시각화**: "결과를 시각화해주세요"

### 직접 실행 (테스트)
```bash
uv run python mcp_server/server.py
```

## 📊 제공되는 도구들

| 도구명 | 설명 | 주요 매개변수 |
|--------|------|---------------|
| `create_sample_data` | 테스트용 샘플 데이터 생성 | filename, n_samples |
| `load_data` | CSV 파일 로드 및 정보 확인 | filepath |
| `preprocess_data` | 데이터 전처리 및 정제 | filepath, target_column |  
| `ridge_analysis` | 릿지 회귀 분석 수행 | filepath, target_column, alpha |
| `visualize_ridge_results` | 결과 시각화 | filepath, target_column, alpha |

## 🔧 개발 및 확장

### 서버 코드 수정
`mcp_server/server.py` 파일을 수정하여 기능을 확장할 수 있습니다.

### 새로운 도구 추가
```python
@mcp.tool()
def new_analysis_tool(param1: str, param2: float) -> dict:
    """새로운 분석 도구 설명"""
    # 분석 로직 구현
    return {"result": "분석 결과"}
```

## 🐛 문제 해결

### 일반적인 문제들
1. **Import 오류**: 패키지가 제대로 설치되었는지 확인
2. **파일 경로 오류**: 절대 경로 사용 권장  
3. **Claude Desktop 연결 실패**: 설정 파일 경로 및 형식 확인
4. **권한 오류**: 파일 읽기/쓰기 권한 확인

### 디버깅
```bash
# MCP 서버 직접 실행하여 오류 확인
uv run python mcp_server/server.py

# Claude Desktop 로그 확인 (Windows)
# %APPDATA%\Claude\logs\mcp*.log
```

## 📁 프로젝트 구조
```
ridge/
├── mcp_server/
│   └── server.py          # MCP 서버 메인 코드
├── pyproject.toml         # 프로젝트 설정
├── claude_desktop_config.json  # Claude Desktop 설정 예시
├── test_guide.md          # 테스트 가이드
└── README.md              # 이 파일
```

## 📚 참고 자료
- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop 다운로드](https://claude.ai/download)
- [scikit-learn Ridge 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

## 🤝 기여하기
1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Submit a pull request

## 📄 라이선스
MIT License - 자유롭게 사용하세요.
