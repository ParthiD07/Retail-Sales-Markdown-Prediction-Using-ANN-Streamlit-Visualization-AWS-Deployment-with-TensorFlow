import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU


# Add a title to your Streamlit app and page configuration

st.set_page_config(page_title= "ANN Predictive Modeling of Retail Sales Price",
                   layout= "wide",initial_sidebar_state='expanded')
st.markdown('<h1 style="color:#fbe337;text-align: center;">ANN Predictive Modeling of Retail Sales </h1>', unsafe_allow_html=True)

df= pd.read_csv("E:/Data Science Project/Final Project/Retail_final_project.csv")
df.drop("Weekly_Sales",axis=1,inplace=True)
# Set up the option menu
menu=option_menu("",options=["Project Overview","ANN Predictive Modeling of Retail Sales Price"],
                        icons=["house","cash"],
                        default_index=1,
                        orientation="horizontal",
                        styles={
        "container": {"width": "100%", "border": "2px ridge", "background-color": "#333333"},
        "icon": {"color": "#FFD700", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#FFFFFF"},
        "nav-link-selected": {"background-color": "#555555", "color": "#FFFFFF"}})

# Set up the information for 'Project Overview' menu
if menu == "Project Overview":
    # Project Title
    st.subheader(':orange[Project Title:]')
    st.markdown('<h5>ANN Predictive Modeling of Retail Sales Price</h5>', unsafe_allow_html=True)
    
    # Problem Statement
    st.subheader(':orange[Problem Statement:]')
    st.markdown('''<h5>Develop a predictive ANN model to forecast department-wide sales for each store over the next year 
                and analyze the impact of markdowns on sales during holiday weeks.Provide actionable insights and 
                recommendations to optimize markdown strategies and inventory management.</h5>''', 
                unsafe_allow_html=True)
    
    # Scope
    st.subheader(':orange[Scope:]')
    st.markdown('''<h5>
                - Data Cleaning and Preparation<br>
                - Feature Engineering<br>
                - Exploratory Data Analysis<br>
                - Deep Learning Algorithms<br>
                - Streamlit Web Application<br>
                - Testing and Validation<br>
                - AWS Deployment<br>
                </h5>''', unsafe_allow_html=True)
    
    # Technologies and Tools
    st.subheader(':orange[Technologies and Tools:]')
    st.markdown('<h5>Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Tensorflow, Keras, Streamlit</h5>', unsafe_allow_html=True)
    
    # Project Learning Outcomes
    st.subheader(':orange[Project Learning Outcomes:]')
    st.markdown('''<h5>
                - Understanding of the end-to-end process of developing a predictive model<br>
                - Practical experience with data collection, preprocessing, and feature engineering<br>
                - Performed exploratory data analysis to identify key business drivers<br>
                - Knowledge of various deep learning algorithms and their applications<br>
                - Skills in evaluating model performance and tuning hyperparameters<br>
                - Familiarity with Streamlit for building interactive web applications<br>
                - Hands-on experience with deploying applications on AWS<br>
                </h5>''', unsafe_allow_html=True)
    
    st.subheader(':orange[Data Sources:]')
    st.markdown('''<h5>Historical sales data for 45 retail stores, covering the period from February 5, 2010, to November 1, 2012. 
                The data includes store-level information, department sales, and markdowns.</h5>''', unsafe_allow_html=True)
    
    
    st.subheader(':orange[Target Audience:]')
    st.markdown('''<h5>Retail managers, data analysts, inventory planners, and business strategists interested in optimizing 
                sales forecasts, markdown strategies, and inventory management.</h5>''', unsafe_allow_html=True)
# User input Values:
class columns:

    Type=['A','B','C']
    Type_encoded={'A':1,'B':2,'C':3}
    Holiday=['False','True']
    Holiday_encoded={'False':0,'True':1}

if menu == "ANN Predictive Modeling of Retail Sales Price":
    st.markdown("<h4 style=color:#fbe337>Enter the following details:",unsafe_allow_html=True)
    st.write('')

    with st.form('price_prediction'):
        col1,col2=st.columns(2)

        with col1:
            Store=st.number_input('**Store**',min_value=1, max_value=45)
            Type=st.selectbox('**Type**',columns.Type)
            Department=st.number_input('**Department**',min_value=1, max_value=100)
            Size=st.selectbox('**Size**',sorted(df['Size'].unique()))
            Day=st.slider('**Day**',min_value=1, max_value=31)
            Month=st.slider('**Month**',min_value=1, max_value=12)
            Week=st.slider('**Week**',min_value=1, max_value=52)
            Year=st.slider('**Year**',min_value=2010, max_value=2024)
            IsHoliday=st.selectbox('**IsHoliday**',columns.Holiday)
        
        with col2:
            Temperature=st.number_input('**Temperature**',min_value=0.1)
            Fuel_Price=st.number_input('**Fuel_Price**',min_value=0.1)
            CPI=st.number_input('**CPI**',min_value=0.1)
            Unemployment=st.number_input('**Unemployment**',min_value=0.1)
            MarkDown1=st.number_input('**Markdown1**',min_value=0.1)
            MarkDown2=st.number_input('**Markdown2**',min_value=0.1)
            MarkDown3=st.number_input('**Markdown3**',min_value=0.1)
            MarkDown4=st.number_input('**Markdown4**',min_value=0.1)
            MarkDown5=st.number_input('**Markdown5**',min_value=0.1)

        st.write('')

        col1,col2,col3 = st.columns([3,5,3])

        with col2:
            button=st.form_submit_button(':green[**Predict Retail Sales Price**]',use_container_width=True)
        
        if button:
            if not all([Store,Type,Department,Size,Day,Month,Week,Year,IsHoliday,Temperature,Fuel_Price,CPI,Unemployment,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5]):
                st.error('Please fill in all the fields')
            else:
                loaded_model = load_model("Retail_model.h5", custom_objects={'LeakyReLU': LeakyReLU})

                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                # Encode the input values
                Type_encoded= columns.Type_encoded[Type]
                Holiday_encoded= columns.Holiday_encoded[IsHoliday]

            #predict the status with regressor model
            user_input=np.array([[Store,Type_encoded,Size,Department,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,Holiday_encoded,Day,Week,Month,Year]])
            # Scale the input values
            scaled_input=scaler.transform(user_input)
            # Make predictions using the trained model
            predict=loaded_model.predict(scaled_input)
            st.markdown(f'<h4 style=color:#fbe337>The Predicted Retail Sales Price is: {predict[0][0]:.2f} </h4>',unsafe_allow_html=True)
            st.snow()





























