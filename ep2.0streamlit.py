import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 加载模型和标准化器
model = load_model('ann_model.h5')
scaler = joblib.load('scaler.pkl') 

# 定义特征选项和名称(保持不变)
Extrauterine_echoes_options = {0: 'Anechoic', 1: 'Homogeneous echo', 2: 'Heterogeneous echo'}

feature_names = [
    "Gestational_age", "Abdominal_tenderness", "Vaginal_bleeding",
    "Pelvic_effusion", "Extrauterine_echoes", "Intrauterine_echoes_size",
    "hCG_ratio", "loghCG_G","Progesterone"
]

# 界面布局
st.title("Risk Stratification System for PUL Patients")

# 用户输入(保持不变)
Gestational_age = st.number_input("Gestational age:", min_value=22, max_value=78, value=42)
Abdominal_tenderness = st.selectbox("Abdominal tenderness:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Vaginal_bleeding = st.selectbox("Vaginal_bleeding:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Pelvic_effusion = st.number_input("Pelvic effusion:", min_value=0, max_value=5.00, value=0.00)
Extrauterine_echoes = st.selectbox("Extrauterine_echoes:", 
                              options=list(Extrauterine_echoes_options.keys()), 
                              format_func=lambda x: Extrauterine_echoes_options[x])
Intrauterine_echoes_size = st.number_input("Intrauterine echoes size:", min_value=0, max_value=5.00, value=0.00)
hCG_ratio = st.number_input("hCG ratio(hCG48h/hCG0h):", min_value=0, max_value=4.5, value=2.0)
loghCG_G = st.number_input("loghCG/G:", min_value=-1, max_value=4, value=1.90)
Progesterone = st.number_input("Progesterone(ng/ml):", min_value=0.2, max_value=60.0, value=15.0)

if st.button("Predict"):
    # 1. 收集所有特征值
    feature_values = {
        "Gestational_age": Gestational_age,
        "Abdominal_tenderness": Abdominal_tenderness,
        "Vaginal_bleeding": Vaginal_bleeding,
        "Pelvic_effusion": Pelvic_effusion,
        "Extrauterine_echoes": Extrauterine_echoes,
        "Intrauterine_echoes_size": Intrauterine_echoes_size,
        "hCG_ratio": hCG_ratio,
        "loghCG_G":loghCG_G,
        "Progesterone": Progesterone
    }
    
    # 2. 创建DataFrame
    input_df = pd.DataFrame([feature_values])
    
    # 3. 仅对连续变量进行标准化(关键修复点)
    continuous_cols = ['Gestational_age', 'Pelvic_effusion','Intrauterine_echoes_size','hCG_ratio','loghCG_G','Progesterone']
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    # 4. 预测概率
    predicted_proba = model.predict_proba(input_df)[0]
    proba_dict = {
    "VIUP": predicted_proba[0],
    "Miscarriage": predicted_proba[1],
    "EP": predicted_proba[2]
    }
    p_viup = proba_dict["VIUP"]
    p_miscarriage = proba_dict["Miscarriage"]
    p_ep = proba_dict["EP"]
    
    # 5. 风险分层逻辑（使用你筛选出的最佳 cutoffs）
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
    # 6. 显示结果
    st.subheader("Prediction Results")
    st.write(f"**Probabilities:**")
    st.metric("Probability: VIUP", f"{p_viup:.1%}")
    st.metric("Probability: Miscarriage", f"{p_miscarriage:.1%}")
    st.metric("Probability: EP", f"{p_ep:.1%}")
    st.success(f"**Risk Level: {risk_stratum}**")
    st.markdown("**Explanation of Risk Levels:**")
    st.markdown("""
    - **Very Low Risk**: Almost certain viable intrauterine pregnancy
    - **Low Risk**: Unlikely to be ectopic pregnancy
    - **Medium Risk**: Uncertain, needs close follow-up
    - **High Risk**: High probability of ectopic pregnancy
    """)
