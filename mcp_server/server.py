from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import json
import os
import tempfile

mcp = FastMCP("ridge-analysis")

# MCP tool: 데이터 로드
@mcp.tool()
def load_data(filepath: str) -> dict:
    """지정된 경로에서 CSV 데이터 로드하고 기본 정보 반환"""
    try:
        if not os.path.exists(filepath):
            return {"error": f"파일을 찾을 수 없습니다: {filepath}"}
        
        df = pd.read_csv(filepath)
        
        return {
            "success": True,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(5).to_dict('records')
        }
    except Exception as e:
        return {"error": f"데이터 로드 중 오류 발생: {str(e)}"}

# MCP tool: 데이터 전처리
@mcp.tool()
def preprocess_data(filepath: str, target_column: str = None) -> dict:
    """결측치 제거 및 수치형 데이터 표준화"""
    try:
        if not os.path.exists(filepath):
            return {"error": f"파일을 찾을 수 없습니다: {filepath}"}
        
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        # 결측치 처리
        df_cleaned = df.dropna()
        
        # 수치형 컬럼만 선택
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column and target_column not in numeric_columns:
            return {"error": f"타겟 컬럼 '{target_column}'이 수치형이 아니거나 존재하지 않습니다"}
        
        # 타겟 컬럼 제외하고 표준화
        feature_columns = [col for col in numeric_columns if col != target_column] if target_column else numeric_columns
        
        if len(feature_columns) > 0:
            scaler = StandardScaler()
            df_cleaned[feature_columns] = scaler.fit_transform(df_cleaned[feature_columns])
        
        # 전처리된 데이터 저장
        processed_filepath = filepath.replace('.csv', '_processed.csv')
        df_cleaned.to_csv(processed_filepath, index=False)
        
        return {
            "success": True,
            "original_shape": original_shape,
            "processed_shape": df_cleaned.shape,
            "processed_filepath": processed_filepath,
            "numeric_columns": numeric_columns,
            "feature_columns": feature_columns,
            "removed_rows": original_shape[0] - df_cleaned.shape[0]
        }
    except Exception as e:
        return {"error": f"데이터 전처리 중 오류 발생: {str(e)}"}

# MCP tool: 릿지 분석
@mcp.tool()
def ridge_analysis(filepath: str, target_column: str, test_size: float = 0.2, alpha: float = 1.0) -> dict:
    """릿지 회귀 분석 수행 및 결과 반환"""
    try:
        import pandas as pd
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        if not os.path.exists(filepath):
            return {"error": f"파일을 찾을 수 없습니다: {filepath}"}
        
        df = pd.read_csv(filepath)
        
        if target_column not in df.columns:
            return {"error": f"타겟 컬럼 '{target_column}'이 존재하지 않습니다"}
        
        # 수치형 데이터만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_column not in numeric_df.columns:
            return {"error": f"타겟 컬럼 '{target_column}'이 수치형이 아닙니다"}
        
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        
        if X.empty:
            return {"error": "특성 데이터가 없습니다"}
        
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 릿지 회귀 모델 훈련
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 계수 정보
        feature_importance = dict(zip(X.columns, model.coef_))
        
        return {
            "success": True,
            "model_params": {"alpha": alpha, "test_size": test_size},
            "coefficients": feature_importance,
            "intercept": model.intercept_,
            "train_metrics": {"mse": train_mse, "r2": train_r2},
            "test_metrics": {"mse": test_mse, "r2": test_r2},
            "feature_names": X.columns.tolist(),
            "data_shape": {"total": df.shape, "features": X.shape, "target": len(y)}
        }
    except Exception as e:
        return {"error": f"릿지 분석 중 오류 발생: {str(e)}"}

# MCP tool: 시각화
@mcp.tool() 
def visualize_ridge_results(filepath: str, target_column: str, alpha: float = 1.0) -> dict:
    """릿지 회귀 결과 시각화 (실제값 vs 예측값 그래프)"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        
        if not os.path.exists(filepath):
            return {"error": f"파일을 찾을 수 없습니다: {filepath}"}
        
        df = pd.read_csv(filepath)
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_column not in numeric_df.columns:
            return {"error": f"타겟 컬럼 '{target_column}'이 수치형이 아니거나 존재하지 않습니다"}
        
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 시각화 생성
        plt.figure(figsize=(10, 6))
        
        # 실제값 vs 예측값 산점도
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title(f'릿지 회귀 예측 결과\n(Alpha={alpha})')
        
        # 잔차 플롯
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('잔차 플롯')
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "success": True,
            "image_base64": img_base64,
            "image_info": "실제값 vs 예측값 및 잔차 플롯"
        }
    except Exception as e:
        return {"error": f"시각화 생성 중 오류 발생: {str(e)}"}

# MCP tool: 샘플 데이터 생성
@mcp.tool()
def create_sample_data(filename: str = "sample_data.csv", n_samples: int = 100) -> dict:
    """릿지 분석 테스트를 위한 샘플 데이터 생성"""
    try:
        import pandas as pd
        import numpy as np
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_features = 5
        
        # 특성 데이터 생성
        X = np.random.randn(n_samples, n_features)
        
        # 타겟 변수 생성 (일부 특성과 선형 관계 + 노이즈)
        true_coef = np.array([2.5, -1.8, 0.5, 3.2, -0.9])
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
        # DataFrame 생성
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # 일부 결측값 추가 (현실적인 데이터)
        missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        missing_col = np.random.choice(feature_names, size=len(missing_idx))
        for idx, col in zip(missing_idx, missing_col):
            df.loc[idx, col] = np.nan
        
        # 파일 저장
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        df.to_csv(filepath, index=False)
        
        return {
            "success": True,
            "filepath": filepath,
            "shape": df.shape,
            "features": feature_names,
            "target": "target",
            "true_coefficients": dict(zip(feature_names, true_coef)),
            "missing_values": df.isnull().sum().to_dict()
        }
    except Exception as e:
        return {"error": f"샘플 데이터 생성 중 오류 발생: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
