import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="CustomerPulse", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #F8F9FA !important; }
* { color: #1a1a1a; }
h1, h2, h3, p, span, div, label { color: #1a1a1a !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a0533 0%, #0d0320 100%) !important; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
.stMetric { background: white !important; border-radius: 10px; padding: 1rem; border-left: 5px solid #7B2FBE; box-shadow: 0 2px 8px rgba(123,47,190,0.15); }
.stMetric label { color: #333333 !important; font-weight: bold !important; }
.stMetric [data-testid="stMetricValue"] { color: #1a0533 !important; font-size: 2rem !important; font-weight: bold !important; }
.insight-box { background: linear-gradient(135deg, #F3E8FF, #EBF5FB); border-radius: 12px; padding: 1.5rem; border-left: 5px solid #7B2FBE; margin-top: 1rem; }
.insight-box p { color: #1a1a1a !important; font-size: 14px !important; line-height: 1.8 !important; }
.section-header { background: linear-gradient(90deg, #1a0533, #7B2FBE); color: white !important; padding: 0.5rem 1rem; border-radius: 8px; margin: 1rem 0 0.5rem 0; font-size: 1rem; letter-spacing: 2px; }
.risk-high { background: #FFE8E8; border-left: 5px solid #CC0000; border-radius: 8px; padding: 1rem; }
.risk-medium { background: #FFF3E0; border-left: 5px solid #FF6600; border-radius: 8px; padding: 1rem; }
.risk-low { background: #E8F5E9; border-left: 5px solid #00AA44; border-radius: 8px; padding: 1rem; }
[data-baseweb="select"] > div { background-color: white !important; border: 2px solid #7B2FBE !important; }
[data-baseweb="select"] * { color: #1a1a1a !important; }
[data-baseweb="popover"] * { color: #1a1a1a !important; background-color: white !important; }
[data-baseweb="menu"] li { color: #1a1a1a !important; background-color: white !important; }
.stSlider label { color: #1a1a1a !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

DATA = 'data/'

@st.cache_data
def load_data():
    customers = pd.read_csv(DATA + 'customers.csv')
    predictions = pd.read_csv(DATA + 'predictions.csv')
    importance = pd.read_csv(DATA + 'feature_importance.csv')
    results = pd.read_csv(DATA + 'model_results.csv')
    return customers, predictions, importance, results

@st.cache_resource
def load_model():
    with open(DATA + 'rf_model.pkl', 'rb') as f:
        return pickle.load(f)

customers, predictions, importance, results = load_data()
model_data = load_model()
rf_model = model_data['model']
lr_model = model_data['lr']
features = model_data['features']

COLORS = ['#7B2FBE','#00AA44','#FF6600','#CC0000','#0066CC','#FF9900','#00AAAA','#FF006E']
CHART = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Georgia', color='#1a1a1a', size=12),
    title_font=dict(size=15, color='#1a1a1a'),
    xaxis=dict(tickfont=dict(color='#1a1a1a', size=11), title_font=dict(color='#1a1a1a'), gridcolor='#F0EBF8'),
    yaxis=dict(tickfont=dict(color='#1a1a1a', size=11), title_font=dict(color='#1a1a1a'), gridcolor='#F0EBF8'),
    legend=dict(font=dict(color='#1a1a1a', size=11))
)

st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem 0; background:white; border-radius:12px; margin-bottom:1rem; box-shadow:0 2px 8px rgba(123,47,190,0.15);'>
    <h1 style='font-size:2.8rem; letter-spacing:8px; color:#1a0533 !important; margin:0; font-family:Georgia,serif;'>CUSTOMERPULSE</h1>
    <p style='color:#7B2FBE !important; letter-spacing:6px; font-size:11px; margin:0.3rem 0;'>CUSTOMER CHURN PREDICTION & RISK INTELLIGENCE PLATFORM</p>
    <p style='color:#555555 !important; font-size:12px; margin:0.3rem 0;'>7,043 Customers | Random Forest + Logistic Regression | Real-Time Risk Scoring</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("NAVIGATE", [
    "Churn Overview",
    "Risk Drivers & EDA",
    "Model Performance",
    "Live Risk Predictor"
])
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:11px; color:#AAAAAA !important;'>CUSTOMERPULSE v1.0<br>ML-Powered Churn Intelligence</p>", unsafe_allow_html=True)

if page == "Churn Overview":
    st.markdown("<h2 style='color:#1a0533 !important; letter-spacing:3px;'>CHURN OVERVIEW</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333333 !important;'>Understanding who is leaving and the financial impact</p>", unsafe_allow_html=True)

    churn_rate = (customers['Churn']=='Yes').mean()
    churned = customers[customers['Churn']=='Yes']
    retained = customers[customers['Churn']=='No']
    monthly_revenue_at_risk = churned['MonthlyCharges'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(customers):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Customers at Risk", f"{len(churned):,}")
    col4.metric("Monthly Revenue at Risk", f"${monthly_revenue_at_risk:,.0f}")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        churn_counts = customers['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn','Count']
        fig = px.pie(churn_counts, values='Count', names='Churn',
                    title="Overall Churn Distribution",
                    color='Churn', color_discrete_map={'Yes':'#CC0000','No':'#00AA44'},
                    hole=0.4)
        fig.update_layout(font=dict(family='Georgia', color='#1a1a1a'),
                         title_font=dict(size=15, color='#1a1a1a'),
                         legend=dict(font=dict(color='#1a1a1a')),
                         height=380, paper_bgcolor='white')
        fig.update_traces(textfont=dict(color='white', size=13))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract_churn = customers.groupby('Contract')['Churn'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        contract_churn.columns = ['Contract','Churn Rate']
        fig2 = px.bar(contract_churn, x='Contract', y='Churn Rate',
                     color='Churn Rate',
                     color_continuous_scale=[[0,'#00AA44'],[0.5,'#FF9900'],[1,'#CC0000']],
                     title="Churn Rate by Contract Type (%)", text='Churn Rate')
        fig2.update_layout(**CHART, height=380, showlegend=False, coloraxis_showscale=False)
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont=dict(color='#1a1a1a'))
        fig2.update_xaxes(title_text="Contract Type")
        fig2.update_yaxes(title_text="Churn Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(customers, x='tenure', color='Churn',
                           color_discrete_map={'Yes':'#CC0000','No':'#7B2FBE'},
                           title="Churn by Customer Tenure (months)",
                           barmode='overlay', opacity=0.7, nbins=30)
        fig3.update_layout(**CHART, height=380)
        fig3.update_xaxes(title_text="Tenure (months)")
        fig3.update_yaxes(title_text="Number of Customers")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.box(customers, x='Churn', y='MonthlyCharges',
                     color='Churn',
                     color_discrete_map={'Yes':'#CC0000','No':'#00AA44'},
                     title="Monthly Charges Distribution by Churn Status")
        fig4.update_layout(**CHART, height=380, showlegend=False)
        fig4.update_xaxes(title_text="Churn Status")
        fig4.update_yaxes(title_text="Monthly Charges ($)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"""<div class='insight-box'><p><b style='color:#1a0533 !important;'>KEY INSIGHT</b><br><br>
    <b>{churn_rate:.1%} of customers are churning</b>, putting <b>${monthly_revenue_at_risk:,.0f} in monthly revenue at risk</b>.
    Month-to-month contract customers churn at over 3x the rate of two-year contract customers.
    Customers in their first 12 months are the highest risk group — early intervention is critical.
    Higher monthly charges correlate strongly with churn — customers on expensive plans need proactive retention outreach.
    </p></div>""", unsafe_allow_html=True)

elif page == "Risk Drivers & EDA":
    st.markdown("<h2 style='color:#1a0533 !important; letter-spacing:3px;'>RISK DRIVERS & ANALYSIS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333333 !important;'>What drives customers to leave?</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(importance.head(10), x='importance', y='feature', orientation='h',
                    color='importance',
                    color_continuous_scale=[[0,'#F3E8FF'],[1,'#1a0533']],
                    title="Top 10 Churn Risk Drivers (Feature Importance)",
                    text='importance')
        fig.update_layout(**CHART, height=420, showlegend=False, coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside',
                         textfont=dict(color='#1a1a1a'))
        fig.update_xaxes(title_text="Feature Importance Score")
        fig.update_yaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        internet_churn = customers.groupby('InternetService')['Churn'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        internet_churn.columns = ['InternetService','Churn Rate']
        payment_churn = customers.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        payment_churn.columns = ['PaymentMethod','Churn Rate']

        fig2 = px.bar(payment_churn.sort_values('Churn Rate', ascending=True),
                     x='Churn Rate', y='PaymentMethod', orientation='h',
                     color='Churn Rate',
                     color_continuous_scale=[[0,'#00AA44'],[0.5,'#FF9900'],[1,'#CC0000']],
                     title="Churn Rate by Payment Method (%)", text='Churn Rate')
        fig2.update_layout(**CHART, height=420, showlegend=False, coloraxis_showscale=False)
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont=dict(color='#1a1a1a'))
        fig2.update_xaxes(title_text="Churn Rate (%)")
        fig2.update_yaxes(title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        senior_churn = customers.groupby('SeniorCitizen')['Churn'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        senior_churn['SeniorCitizen'] = senior_churn['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
        senior_churn.columns = ['Customer Type','Churn Rate']

        security_churn = customers.groupby('OnlineSecurity')['Churn'].apply(
            lambda x: (x=='Yes').mean()*100).reset_index()
        security_churn.columns = ['OnlineSecurity','Churn Rate']

        combined = pd.concat([
            senior_churn.rename(columns={'Customer Type':'Category'}),
            security_churn.rename(columns={'OnlineSecurity':'Category'})
        ])
        fig3 = px.bar(internet_churn, x='InternetService', y='Churn Rate',
                     color='Churn Rate',
                     color_continuous_scale=[[0,'#00AA44'],[0.5,'#FF9900'],[1,'#CC0000']],
                     title="Churn Rate by Internet Service (%)", text='Churn Rate')
        fig3.update_layout(**CHART, height=380, showlegend=False, coloraxis_showscale=False)
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont=dict(color='#1a1a1a'))
        fig3.update_xaxes(title_text="Internet Service")
        fig3.update_yaxes(title_text="Churn Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.bar(senior_churn, x='Customer Type', y='Churn Rate',
                     color='Customer Type',
                     color_discrete_sequence=['#7B2FBE','#CC0000'],
                     title="Churn Rate: Senior vs Non-Senior (%)", text='Churn Rate')
        fig4.update_layout(**CHART, height=380, showlegend=False)
        fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont=dict(color='#1a1a1a'))
        fig4.update_xaxes(title_text="Customer Type")
        fig4.update_yaxes(title_text="Churn Rate (%)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""<div class='insight-box'><p><b style='color:#1a0533 !important;'>KEY INSIGHT</b><br><br>
    The top 3 churn drivers are: Monthly Charges, Total Charges, and Tenure.
    Electronic check users churn significantly more than automatic payment users — likely due to payment friction.
    Fiber optic internet customers churn more than DSL customers despite higher speeds — suggesting a value perception issue.
    Senior citizens churn at a higher rate — they may need dedicated support and simpler plan options.
    </p></div>""", unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown("<h2 style='color:#1a0533 !important; letter-spacing:3px;'>MODEL PERFORMANCE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333333 !important;'>Comparing Logistic Regression vs Random Forest</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    best = results.loc[results['auc_roc'].idxmax()]
    col1.metric("Best Model", best['model'])
    col2.metric("Best Accuracy", f"{results['accuracy'].max():.1%}")
    col3.metric("Best AUC-ROC", f"{results['auc_roc'].max():.3f}")
    col4.metric("Training Data", "5,634 customers")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(results, x='model', y='accuracy',
                    color='model', color_discrete_sequence=['#7B2FBE','#0066CC'],
                    title="Model Accuracy Comparison", text='accuracy')
        fig.update_layout(**CHART, height=380, showlegend=False)
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside',
                         textfont=dict(color='#1a1a1a'))
        fig.update_xaxes(title_text="Model")
        fig.update_yaxes(title_text="Accuracy", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(results, x='model', y='auc_roc',
                     color='model', color_discrete_sequence=['#7B2FBE','#0066CC'],
                     title="AUC-ROC Score Comparison", text='auc_roc')
        fig2.update_layout(**CHART, height=380, showlegend=False)
        fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside',
                          textfont=dict(color='#1a1a1a'))
        fig2.update_xaxes(title_text="Model")
        fig2.update_yaxes(title_text="AUC-ROC Score", range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>RISK SCORE DISTRIBUTION</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = predictions['churn_risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level','Count']
        fig3 = px.pie(risk_counts, values='Count', names='Risk Level',
                     title="Customer Risk Distribution",
                     color='Risk Level',
                     color_discrete_map={'High':'#CC0000','Medium':'#FF9900','Low':'#00AA44'},
                     hole=0.4)
        fig3.update_layout(font=dict(family='Georgia', color='#1a1a1a'),
                          title_font=dict(size=15, color='#1a1a1a'),
                          legend=dict(font=dict(color='#1a1a1a')),
                          height=380, paper_bgcolor='white')
        fig3.update_traces(textfont=dict(color='white', size=13))
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.histogram(predictions, x='rf_prob', color='churn_risk',
                           color_discrete_map={'High':'#CC0000','Medium':'#FF9900','Low':'#00AA44'},
                           title="Churn Probability Distribution",
                           nbins=30, opacity=0.8)
        fig4.update_layout(**CHART, height=380)
        fig4.update_xaxes(title_text="Churn Probability")
        fig4.update_yaxes(title_text="Number of Customers")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""<div class='insight-box'><p><b style='color:#1a0533 !important;'>KEY INSIGHT</b><br><br>
    Both models achieve similar performance — Logistic Regression edges out Random Forest on AUC-ROC for this dataset.
    The model correctly identifies high-risk customers with strong precision.
    AUC-ROC above 0.63 means the model is significantly better than random guessing at identifying churners.
    In a real deployment, this model would allow retention teams to prioritise outreach to the highest-risk customers first.
    </p></div>""", unsafe_allow_html=True)

elif page == "Live Risk Predictor":
    st.markdown("<h2 style='color:#1a0533 !important; letter-spacing:3px;'>LIVE CHURN RISK PREDICTOR</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333333 !important;'>Enter a customer profile to get an instant churn risk score</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='section-header'>CUSTOMER PROFILE</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male","Female"])
        senior = st.selectbox("Senior Citizen", ["No","Yes"])
        partner = st.selectbox("Has Partner", ["Yes","No"])
        dependents = st.selectbox("Has Dependents", ["Yes","No"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 118, 65)

    with col2:
        st.markdown("<div class='section-header'>SERVICES</div>", unsafe_allow_html=True)
        phone = st.selectbox("Phone Service", ["Yes","No"])
        internet = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        security = st.selectbox("Online Security", ["No","Yes","No internet service"])
        tech = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        streaming = st.selectbox("Streaming TV", ["No","Yes","No internet service"])

    with col3:
        st.markdown("<div class='section-header'>CONTRACT & BILLING</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"
        ])
        multiple_lines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])

    total_charges = monthly_charges * tenure

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    input_data = {
        'gender': 1 if gender=='Male' else 0,
        'SeniorCitizen': 1 if senior=='Yes' else 0,
        'Partner': 1 if partner=='Yes' else 0,
        'Dependents': 1 if dependents=='Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone=='Yes' else 0,
        'MultipleLines': {'Yes':2,'No':1,'No phone service':0}[multiple_lines],
        'InternetService': {'Fiber optic':1,'DSL':0,'No':2}[internet],
        'OnlineSecurity': {'Yes':2,'No':1,'No internet service':0}[security],
        'TechSupport': {'Yes':2,'No':1,'No internet service':0}[tech],
        'StreamingTV': {'Yes':2,'No':1,'No internet service':0}[streaming],
        'Contract': {'Month-to-month':0,'One year':1,'Two year':2}[contract],
        'PaperlessBilling': 1 if paperless=='Yes' else 0,
        'PaymentMethod': {'Electronic check':1,'Mailed check':2,'Bank transfer (automatic)':0,'Credit card (automatic)':3}[payment],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'high_value': 1 if monthly_charges > 80 else 0,
        'no_support': 1 if (tech=='No' and security=='No') else 0
    }

    input_df = pd.DataFrame([input_data])[features]
    rf_prob = rf_model.predict_proba(input_df)[0][1]
    lr_prob = lr_model.predict_proba(input_df)[0][1]
    avg_prob = (rf_prob + lr_prob) / 2

    if avg_prob >= 0.6:
        risk_level = "HIGH RISK"
        risk_color = "#CC0000"
        risk_class = "risk-high"
        action = "Immediate retention intervention required. Offer contract upgrade, discount, or dedicated support call."
    elif avg_prob >= 0.35:
        risk_level = "MEDIUM RISK"
        risk_color = "#FF6600"
        risk_class = "risk-medium"
        action = "Monitor closely. Consider proactive outreach with loyalty rewards or service upgrade offer."
    else:
        risk_level = "LOW RISK"
        risk_color = "#00AA44"
        risk_class = "risk-low"
        action = "Customer appears stable. Continue standard engagement and monitor for changes."

    st.markdown("")
    st.markdown(f"""
    <div class='{risk_class}' style='text-align:center; padding:2rem;'>
        <h2 style='color:{risk_color} !important; font-size:2.5rem; margin:0;'>{risk_level}</h2>
        <h3 style='color:{risk_color} !important; font-size:1.8rem; margin:0.5rem 0;'>Churn Probability: {avg_prob:.1%}</h3>
        <p style='color:#333333 !important; font-size:14px; margin:0.5rem 0;'>Random Forest: {rf_prob:.1%} | Logistic Regression: {lr_prob:.1%}</p>
        <p style='color:#1a1a1a !important; font-size:14px; margin-top:1rem;'><b>Recommended Action:</b> {action}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Revenue at Risk", f"${monthly_charges:,.0f}")
    col2.metric("Annual Revenue at Risk", f"${monthly_charges*12:,.0f}")
    col3.metric("Customer Lifetime Value", f"${total_charges:,.0f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_prob*100,
        title={'text': "Churn Risk Score", 'font': {'size': 16, 'family': 'Georgia', 'color': '#1a1a1a'}},
        number={'suffix': "%", 'font': {'size': 24, 'color': '#1a1a1a'}},
        gauge={
            'axis': {'range': [0,100], 'tickcolor': '#1a1a1a'},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0,35], 'color': '#E8F5E9'},
                {'range': [35,60], 'color': '#FFF3E0'},
                {'range': [60,100], 'color': '#FFE8E8'}
            ],
            'threshold': {'line': {'color': risk_color, 'width': 4}, 'thickness': 0.75, 'value': avg_prob*100}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40,b=10,l=20,r=20),
                     paper_bgcolor='white', font=dict(color='#1a1a1a'))
    st.plotly_chart(fig, use_container_width=True)
