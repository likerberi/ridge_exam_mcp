from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
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


def _load_numeric_dataset(filepath: str, target_column: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 존재하지 않습니다")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_column not in numeric_df.columns:
        raise ValueError(f"타겟 컬럼 '{target_column}'이 수치형이 아닙니다")

    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]

    if X.empty:
        raise ValueError("특성 데이터가 없습니다")

    return df, X, y


def _metrics_summary(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _multicollinearity_summary(X: pd.DataFrame) -> dict:
    if X.shape[1] == 1:
        return {
            "condition_number": 1.0,
            "max_feature_correlation": 0.0,
            "mean_abs_correlation": 0.0,
        }

    corr_matrix = X.corr().fillna(0.0)
    off_diagonal = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).stack()
    singular_values = np.linalg.svd(X.to_numpy(), compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1]) if singular_values[-1] != 0 else float("inf")

    return {
        "condition_number": condition_number,
        "max_feature_correlation": float(off_diagonal.abs().max()) if not off_diagonal.empty else 0.0,
        "mean_abs_correlation": float(off_diagonal.abs().mean()) if not off_diagonal.empty else 0.0,
    }


def _fit_model(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model": model,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
        "train_metrics": _metrics_summary(y_train, train_pred),
        "test_metrics": _metrics_summary(y_test, test_pred),
        "coefficients": dict(zip(X_train.columns, [float(value) for value in model.coef_])),
        "intercept": float(model.intercept_),
        "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
    }


def _top_anomalies(indexes, actual_values, predicted_values, top_n: int) -> list[dict]:
    residuals = actual_values - predicted_values
    abs_residuals = np.abs(residuals)
    residual_std = float(abs_residuals.std())
    if residual_std == 0:
        residual_zscores = np.zeros_like(abs_residuals)
    else:
        residual_zscores = (abs_residuals - abs_residuals.mean()) / residual_std

    ranked_positions = np.argsort(abs_residuals)[::-1][:top_n]
    anomalies = []
    for position in ranked_positions:
        anomalies.append({
            "row_index": int(indexes[position]),
            "actual": float(actual_values.iloc[position]),
            "predicted": float(predicted_values[position]),
            "abs_residual": float(abs_residuals[position]),
            "residual_zscore": float(residual_zscores[position]),
        })

    return anomalies

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
        df_cleaned = df.dropna().copy()
        
        # 수치형 컬럼만 선택
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column and target_column not in numeric_columns:
            return {"error": f"타겟 컬럼 '{target_column}'이 수치형이 아니거나 존재하지 않습니다"}
        
        # 타겟 컬럼 제외하고 표준화
        feature_columns = [col for col in numeric_columns if col != target_column] if target_column else numeric_columns
        
        if len(feature_columns) > 0:
            scaler = StandardScaler()
            df_cleaned.loc[:, feature_columns] = scaler.fit_transform(df_cleaned[feature_columns])
        
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
        df, X, y = _load_numeric_dataset(filepath, target_column)

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
            "intercept": float(model.intercept_),
            "train_metrics": {"mse": float(train_mse), "r2": float(train_r2)},
            "test_metrics": {"mse": float(test_mse), "r2": float(test_r2)},
            "feature_names": X.columns.tolist(),
            "data_shape": {"total": df.shape, "features": X.shape, "target": len(y)}
        }
    except Exception as e:
        return {"error": f"릿지 분석 중 오류 발생: {str(e)}"}


@mcp.tool()
def compare_ols_vs_ridge(filepath: str, target_column: str, test_size: float = 0.2, alpha: float = 1.0, top_n: int = 10) -> dict:
    """OLS와 Ridge를 비교하고 상위 이상치 후보를 반환"""
    try:
        _, X, y = _load_numeric_dataset(filepath, target_column)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        ols_result = _fit_model(LinearRegression(), X_train, X_test, y_train, y_test)
        ridge_result = _fit_model(Ridge(alpha=alpha), X_train, X_test, y_train, y_test)

        test_mse_reduction = 0.0
        if ols_result["test_metrics"]["mse"] != 0:
            test_mse_reduction = (
                (ols_result["test_metrics"]["mse"] - ridge_result["test_metrics"]["mse"])
                / ols_result["test_metrics"]["mse"]
            ) * 100

        anomaly_candidates = _top_anomalies(
            y_test.index.to_numpy(),
            y_test.reset_index(drop=True),
            ridge_result["test_predictions"],
            top_n=top_n,
        )

        return {
            "success": True,
            "model_params": {"alpha": alpha, "test_size": test_size, "top_n": top_n},
            "data_summary": {
                "rows": int(len(X)),
                "feature_count": int(X.shape[1]),
                "target_column": target_column,
            },
            "multicollinearity": _multicollinearity_summary(X),
            "ols": {
                "train_metrics": ols_result["train_metrics"],
                "test_metrics": ols_result["test_metrics"],
                "coefficient_l2_norm": ols_result["coefficient_l2_norm"],
                "coefficients": ols_result["coefficients"],
            },
            "ridge": {
                "train_metrics": ridge_result["train_metrics"],
                "test_metrics": ridge_result["test_metrics"],
                "coefficient_l2_norm": ridge_result["coefficient_l2_norm"],
                "coefficients": ridge_result["coefficients"],
            },
            "comparison": {
                "test_mse_reduction_pct": float(test_mse_reduction),
                "test_r2_delta": float(ridge_result["test_metrics"]["r2"] - ols_result["test_metrics"]["r2"]),
                "coefficient_norm_reduction_pct": float(
                    ((ols_result["coefficient_l2_norm"] - ridge_result["coefficient_l2_norm"]) / ols_result["coefficient_l2_norm"]) * 100
                ) if ols_result["coefficient_l2_norm"] else 0.0,
            },
            "review_workflow": {
                "before_manual_review_count": int(len(X_test)),
                "after_prioritized_review_count": int(min(top_n, len(X_test))),
                "review_reduction_pct": float((1 - (min(top_n, len(X_test)) / len(X_test))) * 100),
            },
            "top_anomalies": anomaly_candidates,
        }
    except Exception as e:
        return {"error": f"모델 비교 중 오류 발생: {str(e)}"}

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
    """다중공선성이 있는 회계형 샘플 데이터 생성"""
    try:
        np.random.seed(42)

        revenue = np.random.normal(120_000, 18_000, n_samples)
        receivables = revenue * 0.33 + np.random.normal(0, 2_200, n_samples)
        billed_hours = revenue / 145 + np.random.normal(0, 18, n_samples)
        payroll = billed_hours * 62 + np.random.normal(0, 1_500, n_samples)
        vendor_spend = revenue * 0.18 + payroll * 0.12 + np.random.normal(0, 2_800, n_samples)
        expense_claims = vendor_spend * 0.64 + receivables * 0.11 + np.random.normal(0, 1_400, n_samples)
        travel_cost = expense_claims * 0.42 + np.random.normal(0, 900, n_samples)
        misc_adjustments = np.random.normal(0, 2_000, n_samples)

        anomaly_idx = np.random.choice(n_samples, size=max(3, n_samples // 12), replace=False)
        expense_claims[anomaly_idx] += np.random.normal(12_000, 2_500, len(anomaly_idx))
        misc_adjustments[anomaly_idx] += np.random.normal(4_000, 800, len(anomaly_idx))

        expected_audit_score = (
            revenue * 0.08
            + receivables * 0.21
            + payroll * 0.17
            + vendor_spend * 0.24
            + expense_claims * 0.27
            + travel_cost * 0.06
            + misc_adjustments * 0.12
            + np.random.normal(0, 2_200, n_samples)
        )

        df = pd.DataFrame({
            "revenue": revenue,
            "receivables": receivables,
            "billed_hours": billed_hours,
            "payroll": payroll,
            "vendor_spend": vendor_spend,
            "expense_claims": expense_claims,
            "travel_cost": travel_cost,
            "misc_adjustments": misc_adjustments,
            "expected_audit_score": expected_audit_score,
        })

        missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        feature_names = [
            "revenue",
            "receivables",
            "billed_hours",
            "payroll",
            "vendor_spend",
            "expense_claims",
            "travel_cost",
            "misc_adjustments",
        ]
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
            "target": "expected_audit_score",
            "anomaly_rows": [int(index) for index in anomaly_idx.tolist()],
            "missing_values": df.isnull().sum().to_dict(),
            "note": "매출, 미수금, 급여, 비용청구가 서로 강하게 연동된 회계형 샘플 데이터입니다."
        }
    except Exception as e:
        return {"error": f"샘플 데이터 생성 중 오류 발생: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
