# ============================================
# 2단계 최종 타깃: 전기요금(원)
# 1단계 전력 변수 예측(독립변수 채우기) 후 2단계로 12월 전기요금 예측
# - 입력 파일:
#     ffin_train.csv  (1~11월 학습용)
#     ffin_test.csv   (12월 예측용, id 포함)
#     sample_submission.csv (id, target 템플릿)
# - 출력 파일:
#     submission.csv  (id 기준 target 채운 최종 제출 파일)
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# 1단계 전용 모델
import xgboost as xgb
from catboost import CatBoostRegressor
# (LightGBM은 명시 요구 없음)

# 2단계 최종 모델(전기요금): XGBoost 권장, 없으면 RF로 대체
USE_XGB_STAGE2 = True

# -------------------------------
# 0) 파일 경로
# -------------------------------
TRAIN_PATH = "./data/ffin_train.csv"
TEST_PATH  = "./data/ffin_test.csv"
SAMPLE_SUB = "./data/sample_submission.csv"
OUT_PATH   = "./data/submission_2st_pred.csv"

# -------------------------------
# 1) 데이터 로드 + 기본 피처 생성
# -------------------------------
def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["year"] = df["측정일시"].dt.year
    df["month"] = df["측정일시"].dt.month
    df["day"] = df["측정일시"].dt.day
    df["hour"] = df["측정일시"].dt.hour
    df["minute"] = df["측정일시"].dt.minute
    df["dayofweek"] = df["측정일시"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # 시간 주기 인코딩
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # 피크시간 플래그
    verified_peak_hours = [8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]
    df["is_verified_peak"] = df["hour"].isin(verified_peak_hours).astype(int)

    # 시간대 범주
    def time_cat(h):
        if h in verified_peak_hours: return "verified_peak"
        elif 6 <= h <= 8: return "morning_rush"
        elif 18 <= h <= 21: return "evening_rush"
        elif (22 <= h <= 23) or (0 <= h <= 5): return "night_low"
        else: return "normal"
    df["time_category"] = df["hour"].apply(time_cat)

    return df

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB)

assert "측정일시" in train.columns and "측정일시" in test.columns, "측정일시 컬럼이 필요합니다."
train = create_basic_features(train)
test  = create_basic_features(test)

# 작업유형 없으면 더미 생성
if "작업유형" not in train.columns: train["작업유형"] = "기타"
if "작업유형" not in test.columns:  test["작업유형"]  = "기타"

# 라벨 인코딩(작업유형, time_category) - train+test 결합 기준으로 일관성 보장
le_work = LabelEncoder()
le_time = LabelEncoder()
comb_work = pd.concat([train["작업유형"].astype(str), test["작업유형"].astype(str)], axis=0)
comb_time = pd.concat([train["time_category"].astype(str), test["time_category"].astype(str)], axis=0)
le_work.fit(comb_work)
le_time.fit(comb_time)
train["작업유형_encoded"] = le_work.transform(train["작업유형"].astype(str))
test["작업유형_encoded"]  = le_work.transform(test["작업유형"].astype(str))
train["time_category_encoded"] = le_time.transform(train["time_category"].astype(str))
test["time_category_encoded"]  = le_time.transform(test["time_category"].astype(str))

# 최종 피처(시간/일정/범주 인코딩 위주)
BASIC_FEATS = [
    "year","month","day","hour","minute","dayofweek","is_weekend",
    "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos",
    "is_verified_peak","time_category_encoded","작업유형_encoded"
]
BASIC_FEATS = [c for c in BASIC_FEATS if c in train.columns]

# -------------------------------
# 2) 1단계: 전력 변수 예측(독립변수 채우기)
#    - 타깃별 전용 모델 + 지정 날짜 제거
# -------------------------------
TARGETS_STAGE1 = {
    "전력사용량(kWh)"        : {"model": "RF",   "drop": [("08-01","08-05")]},   # 8/1~8/5
    "지상역률(%)"           : {"model": "XGB",  "drop": [("11-08","11-08")]},   # 11/8
    "진상역률(%)"           : {"model": "CAT",  "drop": [("03-11","03-13")]},   # 3/11~3/13
    "지상무효전력량(kVarh)" : {"model": "RF",   "drop": [("08-01","08-05")]},   # 8/1~8/5
    "진상무효전력량(kVarh)" : {"model": "RF",   "drop": []},                    # 지정 없음
}

def mask_remove_dates_by_md(df: pd.DataFrame, ranges_md: list) -> pd.Series:
    """
    ranges_md: [("MM-DD","MM-DD"), ...]  (동년 가정)
    반환: 제거할 행을 False로 남기는 boolean mask
    """
    m = pd.Series(True, index=df.index)
    months = df["측정일시"].dt.month
    days   = df["측정일시"].dt.day

    for start_md, end_md in ranges_md:
        sm, sd = map(int, start_md.split("-"))
        em, ed = map(int, end_md.split("-"))

        in_range = (months > sm) | ((months == sm) & (days >= sd))
        in_range &= (months < em) | ((months == em) & (days <= ed))
        m &= ~in_range  # 해당 구간은 제거 → mask False
    return m

def fit_and_predict_stage1(train_df, test_df, target_name, spec):
    # 학습 데이터에서 지정 구간 제거
    mask = mask_remove_dates_by_md(train_df, spec["drop"]) if spec["drop"] else pd.Series(True, index=train_df.index)
    tr = train_df.loc[mask].copy()

    X_tr = tr[BASIC_FEATS]
    y_tr = tr[target_name]

    X_te = test_df[BASIC_FEATS]

    model_name = spec["model"]
    if model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=500, max_depth=12, random_state=42, n_jobs=-1
        )
    elif model_name == "XGB":
        model = xgb.XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )
    elif model_name == "CAT":
        model = CatBoostRegressor(
            iterations=800, depth=6, learning_rate=0.05,
            random_state=42, verbose=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_tr, y_tr)
    pred_test = model.predict(X_te)

    # # (선택) 1~11월 내부 검증 MAE를 참고로 계산하고 싶다면 아래 주석 해제
    # # 여기서는 요구사항상 최종 제출만 필요하므로 출력만 가능하게 둠.
    # valid_mask = (train_df["month"] == 11)
    # if valid_mask.any():
    #     pred_valid = model.predict(train_df.loc[valid_mask, BASIC_FEATS])
    #     mae = mean_absolute_error(train_df.loc[valid_mask, target_name], pred_valid)
    #     print(f"[Stage1] {target_name} | model={model_name} | Valid(11월) MAE={mae:.4f}")

    # return pred_test

# 1단계: test에 예측치 컬럼 추가
test_stage1 = test.copy()
for t, spec in TARGETS_STAGE1.items():
    print(f"[Stage1] Predicting {t} using {spec['model']} ...")
    test_stage1[t] = fit_and_predict_stage1(train, test, t, spec)
print("[Stage1] Done.\n")

# -------------------------------
# 3) 2단계: 전기요금(원) 최종 예측
#     - 학습: train의 실제 전력 변수 값 사용
#     - 예측: test에서는 1단계에서 채운 예측치 사용
# -------------------------------
FINAL_TARGET = "전기요금(원)"
assert FINAL_TARGET in train.columns, f"train에 '{FINAL_TARGET}'이 없습니다."

# 2단계 입력 피처 = 기본 피처 + (전력 변수 5개)
POWER_COLS = list(TARGETS_STAGE1.keys())

X_train_final = pd.concat(
    [train[BASIC_FEATS], train[POWER_COLS]],
    axis=1
)
y_train_final = train[FINAL_TARGET]

X_test_final = pd.concat(
    [test_stage1[BASIC_FEATS], test_stage1[POWER_COLS]],
    axis=1
)

# 최종 모델
if USE_XGB_STAGE2:
    try:
        final_model = xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )
    except Exception as e:
        print(f"[Stage2] XGBoost 사용 불가 → RandomForest로 대체 ({e})")
        final_model = RandomForestRegressor(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1)
else:
    final_model = RandomForestRegressor(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1)

print("[Stage2] Training final model for 전기요금(원)...")
final_model.fit(X_train_final, y_train_final)
pred_test_bill = final_model.predict(X_test_final)
pred_test_bill = np.clip(pred_test_bill, a_min=0, a_max=None)

# -------------------------------
# 4) submission.csv 생성
# -------------------------------
assert "id" in test.columns and "id" in sample.columns, "test와 sample_submission에 id 컬럼이 필요합니다."
sub = sample.copy()
sub = sub[["id", "target"]]
sub = sub.merge(test[["id"]], on="id", how="right")  # test id 순서 보장
sub["target"] = pred_test_bill

sub.to_csv(OUT_PATH, index=False)
print(f"[DONE] Saved submission file → {OUT_PATH}")
