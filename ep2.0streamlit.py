import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# 加载模型和标准化器
model = load_model('ann_model.h5')
scaler = joblib.load('scaler.pkl')

# 定义特征选项和名称(保持不变)
Extrauterine_echoes_options = {0: 'Anechoic', 1: 'Homogeneous echo', 2: 'Heterogeneous echo'}

feature_names = [
    "Gestational_age", "Abdominal_tenderness", "Vaginal_bleeding",
    "Pelvic_effusion", "Extrauterine_echoes", "Intrauterine_echoes_size",
    "hCG_ratio", "loghCG_G", "Progesterone"
]

# 页面标题
st.title("Risk Stratification System for PUL Patients")

# 创建左右两栏，宽度比例2:3
left_col, right_col = st.columns([3, 2])

with left_col:
    # 左侧输入区
    Gestational_age = st.number_input("Gestational age:", min_value=22, max_value=78, value=32)
    Abdominal_tenderness = st.selectbox("Abdominal tenderness:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Vaginal_bleeding = st.selectbox("Vaginal bleeding:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    Pelvic_effusion = st.number_input("Pelvic effusion:", min_value=0.00, max_value=5.00, value=1.45)
    Extrauterine_echoes = st.selectbox("Extrauterine echoes:", options=list(Extrauterine_echoes_options.keys()), format_func=lambda x: Extrauterine_echoes_options[x])
    Intrauterine_echoes_size = st.number_input("Intrauterine echoes size:", min_value=0.00, max_value=5.00, value=1.11)
    hCG_ratio = st.number_input("hCG ratio(hCG48h/hCG0h):", min_value=0.00, max_value=4.50, value=1.07)
    loghCG_G = st.number_input("loghCG/G:", min_value=-1.00, max_value=4.00, value=1.48)
    Progesterone = st.number_input("Progesterone(ng/ml):", min_value=0.2, max_value=60.0, value=15.8)

    predict_clicked = st.button("Predict")

with right_col:
    # 右侧结果区，先空着，后面用占位符填充
    result_placeholder = st.empty()

if predict_clicked:
    # 1. 收集输入数据
    feature_values = {
        "Gestational_age": Gestational_age,
        "Abdominal_tenderness": Abdominal_tenderness,
        "Vaginal_bleeding": Vaginal_bleeding,
        "Pelvic_effusion": Pelvic_effusion,
        "Extrauterine_echoes": Extrauterine_echoes,
        "Intrauterine_echoes_size": Intrauterine_echoes_size,
        "hCG_ratio": hCG_ratio,
        "loghCG_G": loghCG_G,
        "Progesterone": Progesterone
    }

    input_df = pd.DataFrame([feature_values])

    # 2. 连续变量标准化
    continuous_cols = ['Gestational_age', 'Pelvic_effusion', 'Intrauterine_echoes_size', 'hCG_ratio', 'loghCG_G', 'Progesterone']
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    # 3. One-Hot 编码 Extrauterine_echoes
    extrauterine_encoded = np.zeros(3)
    extrauterine_encoded[Extrauterine_echoes] = 1

    # 4. 拼接最终模型输入
    final_input = np.concatenate([
        input_df[continuous_cols].values.flatten(),                # 6个连续变量
        [Abdominal_tenderness], [Vaginal_bleeding],               # 2个二分类变量
        extrauterine_encoded                                      # 3个One-Hot变量
    ]).reshape(1, -1)  # 总共11个特征

    # 5. 预测概率
    predicted_proba = model.predict(final_input)[0]
    proba_dict = {
        "VIUP": predicted_proba[0],
        "Miscarriage": predicted_proba[1],
        "EP": predicted_proba[2]
    }
    p_viup = proba_dict["VIUP"]
    p_miscarriage = proba_dict["Miscarriage"]
    p_ep = proba_dict["EP"]

    # 6. 风险分层逻辑
    def stratify(p_intra, p_miscarriage, p_ep,
                 p_intra_cutoff=0.9,
                 p_total_cutoff=0.8,
                 p_ep_cutoff_low=0.03,
                 p_ep_cutoff_high=0.15):
        if p_intra >= p_intra_cutoff:
            return "Very Low Risk"
        elif (p_intra + p_miscarriage >= p_total_cutoff) and (p_ep < p_ep_cutoff_low):
            return "Low Risk"
        elif p_ep >= p_ep_cutoff_high:
            return "High Risk"
        else:
            return "Medium Risk"

    risk_stratum = stratify(p_viup, p_miscarriage, p_ep)

    # 7. 显示结果到右侧占位符
    with result_placeholder.container():
        st.subheader("Prediction Results")
        st.write(f"**Probabilities:**")
        st.metric("Probability: VIUP", f"{p_viup:.1%}")
        st.metric("Probability: Miscarriage", f"{p_miscarriage:.1%}")
        st.metric("Probability: EP", f"{p_ep:.1%}")
        if risk_stratum in ["Very Low Risk", "Low Risk"]:
            st.success(f"**Risk Level: {risk_stratum}**")    # 绿色
        elif risk_stratum == "Medium Risk":
            st.warning(f"**Risk Level: {risk_stratum}**")    # 橙色
        elif risk_stratum == "High Risk":
            st.error(f"**Risk Level: {risk_stratum}**")      # 红色，更显眼，也可以改成 success 绿色
        else:
            st.info(f"**Risk Level: {risk_stratum}**")       # 其他情况蓝色提示
        st.markdown("**Explanation of Risk Levels:**")
        st.markdown("""
        - **Very Low Risk**: Almost certain viable intrauterine pregnancy
        - **Low Risk**: Unlikely to be ectopic pregnancy
        - **Medium Risk**: Uncertain, needs close follow-up
        - **High Risk**: High probability of ectopic pregnancy
        """)

