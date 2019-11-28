import base64
import datetime
import io
import json
import warnings

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from lifelines.utils import concordance_index as C_index



xvar_list = ['Gender', 'Age_group', 'Smoker', 'BMI_group', 'CHr_group','Diab_Gly_group', 'BP_group', 'Dis_Histo']
gender_type = CategoricalDtype(categories=['Female', 'Male'], ordered=True)
age_type = CategoricalDtype(categories=['[18,30]', '(30,40]','(40,50]',  '(50,60]', '(60,70]'], ordered=True)
smoker_type = CategoricalDtype(categories=['Non Smoker','Smoker'], ordered=True)
bmi_type = CategoricalDtype(categories=['(18.5,33]', '(33,37]', '(37,40]', '(40,43]', '(43,Inf]'], ordered=True)
chr_type = CategoricalDtype(categories=['[0,6]', '(6,8]', '(8,15]'], ordered=True)
diab_type = CategoricalDtype(categories=['No_(0,6]', 'No_(6,10.5]', 'Yes_(0,7]', 'Yes_(7,8.5]','Yes_(8.5,10.5]','Gly_abn'], ordered=True)
bp_type = CategoricalDtype(categories=['Normal', 'PP_abn', 'S1_HT', 'S2_HT'], ordered=True)
dishis_type = CategoricalDtype(categories=['No', 'Yes_S', 'Yes_H', 'Yes_HS'], ordered=True)

data_init = pd.read_csv("toy_data.csv")
data_init.Gender = data_init.Gender.astype(gender_type)
data_init.Age_group = data_init.Age_group.astype(age_type)
data_init.Smoker = data_init.Smoker.astype(smoker_type)
data_init.BMI_group = data_init.BMI_group.astype(bmi_type)
data_init.CHr_group = data_init.CHr_group.astype(chr_type)
data_init.Diab_Gly_group = data_init.Diab_Gly_group.astype(diab_type)
data_init.BP_group = data_init.BP_group.astype(bp_type)
data_init.Dis_Histo = data_init.Dis_Histo.astype(dishis_type)


def logitreg_fit(data_surv, xvar_list, test=None):
    data_raz = pd.DataFrame(np.repeat(data_surv.values,np.ceil(data_surv.Surv/12).astype("int64"),axis=0),columns=data_surv.columns)
    FUY_list = []
    Death_list = []
    for seqn, group in data_raz.groupby("SEQN",sort=False):
        FUY_list = np.concatenate((FUY_list,np.arange(len(group)))).astype("int")
        Death_list = np.concatenate((Death_list,list(np.zeros(len(group)-1))+list(np.unique(group.Death)))).astype("int")
    data_raz["FUY"] = FUY_list
    data_raz["Expo"] = np.minimum(data_raz["Surv"]/12-data_raz["FUY"],1)
    data_raz["Death"] = Death_list

    X_train = data_raz.loc[:,xvar_list+["FUY"]]
    y_train = data_raz.Death
    X_train = pd.get_dummies(X_train)
    X_train.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                  'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
    #logitreg = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial(),freq_weights=data_raz.WEIGHTS.astype("float"))
    logitreg = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())
    logitreg_fit = logitreg.fit()

    ispldata_raz = pd.DataFrame(np.repeat(data_surv.values,max(data_raz.FUY)+1,axis=0),columns=data_surv.columns)
    ispldata_raz["FUY"]=list(range(max(data_raz.FUY)+1))*len(data_surv)
    X_ispl = ispldata_raz.loc[:,xvar_list+["FUY"]]
    X_ispl = pd.get_dummies(X_ispl)
    col_tmp = X_ispl.columns
    for ele in np.setdiff1d(X_ispl.columns,col_tmp):
        X_ispl[ele] = 0
    X_ispl.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                  'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
    qx_train = logitreg_fit.predict(sm.add_constant(X_ispl))
    e0_train = np.sum((1-qx_train).values.reshape(-1,17),axis=1)
    cindex_train = round(C_index(data_surv.Surv, e0_train, data_surv.Death),5)
    
    if test is not None:
        testdata = test
        testdata_raz = pd.DataFrame(np.repeat(testdata.values,max(data_raz.FUY)+1,axis=0),columns=testdata.columns)
        testdata_raz["FUY"]=list(range(max(data_raz.FUY)+1))*len(testdata)
        X_test = testdata_raz.loc[:,xvar_list+["FUY"]]
        X_test = pd.get_dummies(X_test)
        col_tmp = X_test.columns
        for ele in np.setdiff1d(X_train.columns,col_tmp):
            X_test[ele] = 0
        X_test.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                      'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
        qx_test = logitreg_fit.predict(sm.add_constant(X_test))
        e0_test = np.sum((1-qx_test).values.reshape(-1,17),axis=1)
        cindex_test = round(C_index(testdata.Surv, e0_test, testdata.Death),5)
    else:
        cindex_test = np.nan
    
    return [cindex_train,cindex_test, logitreg_fit]

def poisreg_fit(data_surv, xvar_list, test=None):
    data_raz = pd.DataFrame(np.repeat(data_surv.values,np.ceil(data_surv.Surv/12).astype("int64"),axis=0),columns=data_surv.columns)
    FUY_list = []
    Death_list = []
    for seqn, group in data_raz.groupby("SEQN",sort=False):
        FUY_list = np.concatenate((FUY_list,np.arange(len(group)))).astype("int")
        Death_list = np.concatenate((Death_list,list(np.zeros(len(group)-1))+list(np.unique(group.Death)))).astype("int")
    data_raz["FUY"] = FUY_list
    data_raz["Expo"] = np.minimum(data_raz["Surv"]/12-data_raz["FUY"],1)
    data_raz["Death"] = Death_list

    X_train = data_raz.loc[:,xvar_list+["FUY"]]
    y_train = data_raz.Death
    X_train = pd.get_dummies(X_train)
    X_train.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                  'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
    #poisreg = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Poisson(),freq_weights=data_raz.WEIGHTS.astype("float"))
    poisreg = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Poisson())
    poisreg_fit = poisreg.fit()
    
    ispldata_raz = pd.DataFrame(np.repeat(data_surv.values,max(data_raz.FUY)+1,axis=0),columns=data_surv.columns)
    ispldata_raz["FUY"]=list(range(max(data_raz.FUY)+1))*len(data_surv)
    X_ispl = ispldata_raz.loc[:,xvar_list+["FUY"]]
    X_ispl = pd.get_dummies(X_ispl)
    col_tmp = X_ispl.columns
    for ele in np.setdiff1d(X_ispl.columns,col_tmp):
        X_ispl[ele] = 0
    X_ispl.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                  'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
    qx_train = poisreg_fit.predict(sm.add_constant(X_ispl))
    e0_train = np.sum((1-qx_train).values.reshape(-1,17),axis=1)
    cindex_train = round(C_index(data_surv.Surv, e0_train, data_surv.Death),5)
    
    if test is not None:
        testdata = test
        testdata_raz = pd.DataFrame(np.repeat(testdata.values,max(data_raz.FUY)+1,axis=0),columns=testdata.columns)
        testdata_raz["FUY"]=list(range(max(data_raz.FUY)+1))*len(testdata)
        X_test = testdata_raz.loc[:,xvar_list+["FUY"]]
        X_test = pd.get_dummies(X_test)
        col_tmp = X_test.columns
        for ele in np.setdiff1d(X_train.columns,col_tmp):
            X_test[ele] = 0
        X_test.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                      'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
        qx_test = poisreg_fit.predict(sm.add_constant(X_test))
        e0_test = np.sum((1-qx_test).values.reshape(-1,17),axis=1)
        cindex_test = round(C_index(testdata.Surv, e0_test, testdata.Death),5)
    else:
        cindex_test = np.nan
    
    return [cindex_train,cindex_test, poisreg_fit]

def var_class_table(category_pdSeries):
    summary_table = category_pdSeries.value_counts(sort=False)
    return {"x": list(summary_table.index), "y": list(summary_table.values)}

def death_proba(df, UW_year, xvar):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freq_table = df[xvar][(df.Surv<=UW_year*12)*(df.Death==1)].value_counts(sort=False)/df[xvar].value_counts(sort=False)
    return {"x": list(freq_table.index), "y":list(freq_table.values)}

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    dict_tmp = {"data_tmp": df, "filename": filename}
    return  dict_tmp

def tmpdata2dashtable(df,filename):
    return html.Div([
        html.H6(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={'maxHeight': '300px'},
            style_cell={'minWidth': '20px', 'maxWidth': '50px'}
        ),
        html.Hr()  # horizontal line
    ])


def modeling_launch(values, data, percent):
    res_list = []
    data_train, data_test = train_test_split(data, test_size = percent, random_state=69)
    for mdl in values:
        if mdl == "Logistic Model":
            res = logitreg_fit(data_train,xvar_list,data_test)
            res_list.append({"Model": mdl, "Training": res[0], "Test": res[1]})
        elif mdl == "Poisson Model":
            res = poisreg_fit(data_train, xvar_list,data_test)
            res_list.append({"Model":mdl, "Training": res[0], "Test": res[1]})
        else:
            res_list.append({"Model":mdl,"Training":np.nan, "Test": np.nan})
    return pd.DataFrame(res_list)


def mapping1(value,xvar):  # for age, bmi, chr (value: number)
    categ_dict = {"age":['[18,30]', '(30,40]','(40,50]',  '(50,60]', '(60,70]'],
                  "bmi":['(18.5,33]', '(33,37]', '(37,40]', '(40,43]', '(43,Inf]'],
                  "chr":['[0,6]', '(6,8]', '(8,15]']}
    bins_dict = {"age":[17,30,40,50,60,70],
                  "bmi":[18.4,33,37,40,43],
                  "chr":[0,6,8,15]}
    return categ_dict[xvar][np.digitize(value, bins_dict[xvar], True)-1]

def mapping2(value1, value2, xvar): # for (diabetes,glycohemoglobin) and (SBP, DBP) (value1, value2: number or string)
    if xvar == "diab-gly":
        if value1 == "Diabetic":
            bins = [0,7,8.5,10.5]
            categ = ['Yes_(0,7]', 'Yes_(7,8.5]','Yes_(8.5,10.5]',"Gly_abn"]
            return categ[np.digitize(value2, bins, True)-1]
        else:
            bins = [0,6,10.5]
            categ = ['No_(0,6]', 'No_(6,10.5]',"Gly_abn"]
            return categ[np.digitize(value2, bins, True)-1]
    elif xvar == "bp":
        if value1 <= 90 and value2<= 60:
            return "PP_abn"
        else:
            if value1<=140 and value2<=90:
                return "Normal"
            else: 
                if value1<=160 and value2<=100:
                    return "S1_HT"
                else:
                    return "S2_HT"

def mapping3(L): # for disease history (L: list)
    if ("Stroke" in L) and ("Heart Disease" in L):
        return 'Yes_HS'
    else:
        if "Stroke" in L:
            return "Yes_S"
        elif "Heart Disease" in L:
            return "Yes_H"
        else:
            return "No"

        
def mapping_casestudy(data_ref,xvar_list, age, gender, smoker, h, w, sbp, dbp, diab, gly, tc , hdl, diss):
    bmi = w**2/h
    ch_r = tc/hdl
    Gender = [gender,gender]
    Age_group = [mapping1(age,"age"),mapping1(age,"age")]
    Smoker = [smoker, smoker]
    BMI_group = [mapping1(bmi,"bmi"),"(18.5,33]"]
    CHr_group = [mapping1(ch_r,"chr"), "(0,6]"]
    Diab_Gly_group = [mapping2(diab,gly,"diab-gly"),'No_(0,6]']
    BP_group = [mapping2(sbp,dbp,"bp"), "Normal"]
    Dis_Histo = [mapping3(diss), "No"]
    df_tmp = pd.DataFrame({"Gender":Gender, "Age_group":Age_group, "Smoker":Smoker, "BMI_group":BMI_group,
                           "CHr_group":CHr_group, "Diab_Gly_group":Diab_Gly_group, "BP_group":BP_group, "Dis_Histo":Dis_Histo},
                          index=["case study","standard"])
    df_tmp = pd.get_dummies(df_tmp)
    df_ref = data_ref.loc[:,xvar_list]
    df_ref = pd.get_dummies(df_ref)
    col_tmp = df_tmp.columns
    for ele in np.setdiff1d(df_ref.columns,col_tmp):
        df_tmp[ele] = 0
    df_tmp.drop(["Gender_Female","Age_group_[18,30]","Smoker_Non Smoker",'BMI_group_(18.5,33]', 'CHr_group_[0,6]','Diab_Gly_group_No_(0,6]',
                 'BP_group_Normal','Dis_Histo_No'],axis=1, inplace=True)
    return df_tmp


def casestudy_results(df, selected_model,max_FUY = 17):
    df = pd.DataFrame(np.repeat(df.values,max_FUY,axis=0),columns=df.columns)
    df["FUY"] = list(range(max_FUY))*2
    qx = selected_model.predict(sm.add_constant(df))
    Sx = np.cumprod((1-qx.values).reshape(2,-1),axis=1)
    return {
        "data": [
                        {
                            "x" : list(range(max_FUY+1)),
                            "y" : [1]+list(Sx[0]),
                            "name":'Case Study',
                            "marker":{
#                                "color": "rgb(55, 83, 109)"
                            }
                        },
                        {
                            "x" : list(range(max_FUY+1)),
                            "y" : [1]+list(Sx[1]),
                            "name" : "Standard",
                            "marker": {
 #                               "color" : "rgb(26, 118, 255)"
                            }
                        }
                    ],
            "layout": {
                "title" : {"text":'Survival Curves', "font":{"size":20}},
                "showlegend" : True,
                "legend" : {
                    "x" : 1.0,
                    "y" : 1.0
                },
                "xaxis" : {"title": "Follow-up Year"},
                "yaxis" : {"title": "Survival Probability"},
                "margin" : {"l":"40px", "r":"0px", "t":"40px", "b":"30px"}
            }
        }

            



"""""""""

APP layout

"""""""""


data_upload_dict = {"data":data_init,"filename":"","max_FUY":50}
candidate_models_dict = {}
final_model = {}
#n_clickss = {"n":0}
style_DivStep2 = {'width':"100%","margin":"20px","display":"none"}
style_DivStep3 = {'width':"100%","margin":"20px","display":"none"}
style_DivStep3_report = {'width':"100%","margin":"10px","display":"none"}
style_DivStep4 = {'width':"100%","margin":"20px","display":"none"}
style_DivStep4_survcurve = {'width':"70%","marginTop":"30px","paddingLeft":"12%","paddingRight":"12%","display":"none"}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(id="hidden-div", style={"display":"none"}),
    
    ## Part 1 . Upload Data
    html.Div(id='Div-Upload-Data', 
             children=[
                 html.H1("Step 1. Data Upload"),
                 html.H6("Please upload survival data (only .csv can be accepted): "),
                 dcc.Upload(id='upload-data',
                            children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={
                                'width': '100%','height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed',
                                'borderRadius': '5px','textAlign': 'center','margin': '10px'}),
                 html.Div([html.Button("Upload!", id="Button-data-upload", 
                                       style={ "margin": "10px",'textAlign': 'center',"display":"inline-block",
                                              "backgroundColor":'#FCDA44'}),
                           html.Small(id="Text-data-upload",style={"display":"inline-block","textAlign":"center"})]),
                 html.Br(),
                 html.Div(id='dashtable-data-upload')#,style={"height":"400px"})
             ]
            ),


    ## Part 2.  Descriptive Statistics
    html.Div(id="Div-Descrpt-Stats",
             children=[
                 html.H1('Step 2. Descriptive Statistics'),
                 html.Header("Select one risk factor to check the descriptive statistics:"),
                 dcc.Dropdown(
                     id="xvar-dropdown",
                     options = [{"label": xvar, "value": xvar} for xvar in xvar_list],
                     value = "Gender",
                     style={'width': '50%',"margin": "5px"}
                 ),
                 html.Div([
                     dcc.Graph(id="graph-xvar-dropdown"),
                     dcc.Slider(
                         id='UWY-slider',
                         min=0,
                         max=50,
                         value=0,
                         step = 0.1,
                         marks={str(year): str(year) for year in range(1+50)},
                         updatemode = "drag"
                     )],
                     style={'width': '70%',"margin": "1px", "paddingLeft":"15%", "paddingRight":"15%"}
                 ),
                 html.Br(),
             ],style=style_DivStep2
            ),
    
    ## Part 3.  Modeling
    html.Div(id = 'Div-Modeling', 
             children = [
                 html.H1("Step 3. Modeling"),
                 html.Div([                     
                     html.Header(html.Strong("Determine candidate models:"),style={"margin": "0px", "verticalAlign":"middle"}),
                     dcc.Dropdown(
                         id = "Dropdown-candidate-models",
                         options=[
                             {'label': 'Poisson Model', 'value': 'Poisson Model'},
                             {'label': 'Logistic Model', 'value': 'Logistic Model'},
                             {'label': 'Random Survival Forest', 'value': 'Random Survival Forest'}
                         ],
                         value="",
                         multi = True,
                         style = {'width': '90%', "margin": "5px"}
                     )],style = {'width': '25%',"margin": "10px","marginLeft": "15px", "display":"inline-block",'verticalAlign': 'middle'}),

                 html.Div(children = [
                     html.Header(html.Strong("Folds number of cross validation:"),style={"margin": "0px", "verticalAlign":"middle"}),
                     dcc.Input(
                         id = "Input-CVfolds",
                         placeholder = 'Enter a number...',
                         type = 'number',
                         value = 5,
                         style = {'width': '50%', "margin": "0px"}
                     )
                 ], style = {'width': '20%',"margin": "10px","display":"inline-block",'verticalAlign': 'middle'}),
                 
                 html.Div(children = [
                     html.Header(html.Strong("Training/Test data split percent:"),style={"margin": "0px", "verticalAlign":"middle"}),
                     dcc.Slider(
                         id = "Slider-textdata-split-percent",
                         min = 0,
                         max = 100,
                         value = 20,
                         step = 1,
                         marks={str(i): "{}%".format(i) for i in 10*(np.arange(11))}
                     )
                 ], style = {'width': '30%',"margin": "10px", "marginLeft":"5%", 
                             "display":"inline-block",'verticalAlign': 'middle',"paddingBottom":"15px"}),

                 html.Button("Go for modeling!", id = "Button-modeling", 
                             style = { "margin": "5px","marginLeft": "25px","backgroundColor":'#FCDA44'}),
                 html.Div([
                     html.Div(id="Dashtable-model-performance"),
                     html.Div(id = "Div-dropdown-model-select"),
                     html.Div([
                         html.Strong("So make your final choice on model:", 
                                     style={"margin": "5px", "display":"inline-block", "verticalAlign":"middle"}),
                        dcc.Dropdown(
                            id="Dropdown-model-select",
                            style={"width": "60%","margin": "5px", "display":"block", "verticalAlign":"middle"},
                            value=None,
                            options = []
                         ),
                        html.Button("Submit", id="Button-model-determine", 
                                    style = { "margin": "10px","backgroundColor":'#FCDA44',"display":"block"})
                     ],id="Div-reporting",style = style_DivStep3_report)
                 ])
             ]
            ),
        
    ## Part 4. Case Study
    html.Div(id = 'Div-Case-Study', 
             children = [
                 html.H1("Step 4. Case Study"),
                 html.Div([
                     html.Div([
                         html.Div([
                             html.Strong("Age:", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-age",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"30%","marginLeft":"15px","marginRight":"15px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Gender:", style = {"margin":"5px","display":"block"}),
                             dcc.Dropdown(id="Input-gender", options=[{"label": ele, "value":ele} for ele in ["Male", "Female"]],
                                          value=None,style = {"margin":"5px","display":"block"}),
                         ],style={"width":"30%","marginLeft":"15px","marginRight":"15px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Smoking status:", style = {"margin":"5px","display":"block"}),
                             dcc.Dropdown("Input-smoke" ,options=[{"label": ele, "value":ele} for ele in ["Smoker", "Non Smoker"]],
                                          value=None,style = {"margin":"5px","display":"block"})
                         ],style={"width":"30%","marginLeft":"15px","marginRight":"15px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                     ],style = {"width":"100%", "margin":"10px"}),
                     
                     html.Div([
                         html.Div([
                             html.Strong("Height(cm):", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-height",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Weight(kg):", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-weight",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Systolic Blood Pressure:", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-sbp",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Diastolic Blood Pressure:", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-dbp",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'})
                     ],style = {"width":"100%", "margin":"10px"}),
                     
                     html.Div([
                         html.Div([
                             html.Strong("Diabetes:", style={"margin":"5px","display":"block"}),
                             dcc.Dropdown("Input-diab", options=[{"label": ele, "value":ele} for ele in ["Diabetic", "Non Diabetic"]],
                                          value=None,style = {"margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Glycohemoglobin(%):", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-glyco",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("Total Cholesterol(mmol/L):", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-tc",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'}),
                         html.Div([
                             html.Strong("HDL-C(mmol/L):", style={"margin":"5px","display":"block"}),
                             dcc.Input(id = "Input-hdl",placeholder = 'Enter a number...',type = 'number',value = None,
                                      style={"width":"100%","margin":"5px","display":"block"}),
                         ],style={"width":"20%","marginLeft":"28px","marginRight":"28px","display":"inline-block",
                                  'verticalAlign': 'middle'})
                     ],style = {"width":"100%", "margin":"10px"}),                     
                     
                     html.Div([
                         html.Strong("Disease History:", style={"marginLeft":"30px","display":"inline-block"}),
                         dcc.Checklist(id="Input-dishis",options=[{"label": ele, "value":ele} for ele in ["Disease History", "Stroke"]],
                                       value=[],style = {"margin":"5px","display":"inline-block"},
                                       labelStyle={'display': 'inline-block',"marginLeft":"10px", "font":"bold"}),
                         html.Button("Calculate!", id="Button-calculator", 
                                    style = { "marginLeft": "50%","backgroundColor":'#FCDA44',"display":"inline-block"})
                     ],style = {"width":"100%", "margin":"20px"}),
                     
                 ],style = {"width":"100%", "margin":"5px","backgroundColor":"#c5d2f1", "display":"inline-block"}
                 ), #6984fc
                 
                 html.Div(id="Div-casestudy-results",children = [
                     dcc.Graph(id='Graph-survival-curves',style={'height': "500px", "marginTop":"20px"})
                 ], style = style_DivStep4_survcurve)
             ],style = style_DivStep4
            )
    
##  End   
])


@app.callback([Output('dashtable-data-upload', 'children'),
               Output("Text-data-upload", "children"),
#               Output("UWY-slider", "min"),
               Output("UWY-slider","max"),
               Output("UWY-slider","marks"),
               Output("Div-Descrpt-Stats","style"),
               Output("Div-Modeling","style")],
              [Input("Button-data-upload","n_clicks")],
              [State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def update_loaddata(n_clicks, content, name):
    #n_clickss.update({"n":n_clicks})
    children_dashtable = ""
    children_text = ""
    if content is not None:
        dict_tmp = parse_contents(content, name)
        df_tmp = dict_tmp["data_tmp"]
        filename_tmp = dict_tmp["filename"]
        children_dashtable = tmpdata2dashtable(df_tmp, filename_tmp)        
        children_text = "Uploaded successfully!"
        style_DivStep2.update({"display":"inline-block"})
        style_DivStep3.update({"display":"inline-block"})
        
        try:
            df_tmp.Gender = df_tmp.Gender.astype(gender_type)
            df_tmp.Age_group = df_tmp.Age_group.astype(age_type)
            df_tmp.Smoker = df_tmp.Smoker.astype(smoker_type)
            df_tmp.BMI_group = df_tmp.BMI_group.astype(bmi_type)
            df_tmp.CHr_group = df_tmp.CHr_group.astype(chr_type)
            df_tmp.Diab_Gly_group = df_tmp.Diab_Gly_group.astype(diab_type)
            df_tmp.BP_group = df_tmp.BP_group.astype(bp_type)
            df_tmp.Dis_Histo = df_tmp.Dis_Histo.astype(dishis_type)
        except Exception as e:
            print(e)
            return {}
        data_upload_dict.update({"data":df_tmp, "filename":filename_tmp})
    df_tmp = data_upload_dict["data"]
    #min_surv_year = int(np.floor(df_tmp.Surv.min()/12))
    max_surv_year = int(np.ceil(df_tmp.Surv.max()/12))
    data_upload_dict.update({"max_FUY":max_surv_year, })
    marks_tmp = {str(year): str(year) for year in range(1+max_surv_year)}
    
    return  children_dashtable, children_text, max_surv_year, marks_tmp, style_DivStep2, style_DivStep3             

    
@app.callback(
    Output(component_id = "graph-xvar-dropdown", component_property = "figure"),
    [Input(component_id = "xvar-dropdown", component_property = "value"),
     Input(component_id = "UWY-slider", component_property = "value")]
)
def update_figure(selected_xvar, selected_UWY):
    data = data_upload_dict["data"]
    max_surv_year = data_upload_dict["max_FUY"]
    y2_up = 0.1+max(death_proba(data, max_surv_year, selected_xvar)["y"])
    
    return {"data":[{
                "type": "bar",
                "x": var_class_table(data[selected_xvar])["x"],
                "y": var_class_table(data[selected_xvar])["y"],
                "name" : "Sampling numbers",
                "text": "",
                "opacity" : 0.7
            },{
                "type": "scatter",
                "x": death_proba(data,selected_UWY,selected_xvar)["x"],
                "y": death_proba(data,selected_UWY,selected_xvar)["y"],
                "name": "Empirical mortality",
                "marker": {
                    'size': 10,
                    'opacity': 0.8
                },
                "line":{"opacity":0.8},
                "yaxis": "y2",
            }],
            "layout":{
                "title": "empirical mortality on different classes of " + selected_xvar,
                "xaxis": {"title": selected_xvar, "showgrid":False},
                "yaxis": {"title": "Frequence", "showgrid":False},
                "yaxis2": {"title": "Percentage", "overlaying": 'y', "side": 'right', "range": [0,y2_up], "showgrid":False},
                'legend': dict(orientation='v',yanchor='top',xanchor='right',y=1.1,x=1.2)
            }
        }


@app.callback(
    [Output(component_id="Dashtable-model-performance", component_property="children"),
    Output(component_id="Dropdown-model-select", component_property="options"),
    Output("Div-reporting", "style")],
    [Input(component_id = "Button-modeling", component_property="n_clicks")],
    [State(component_id = "Dropdown-candidate-models", component_property = "value"),
     State("Slider-textdata-split-percent", "value")]
)
def update_modeling_candidates(n_clicks, values, size):
    if n_clicks is None:
        raise PreventUpdate
    else:
        style_DivStep3_report.update({"display":"block"})
        data_train = data_upload_dict["data"]
        df_res = modeling_launch(values, data_train, size/100)
        children_dashtable = html.Div([
            html.Br(),
            html.Header("Performance Report according to Concordence Index: ", style={"margin":"5px"}),
            dash_table.DataTable(
            data=df_res.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_res.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={'maxHeight': '300px'}
            #style_cell={'minWidth': '20px', 'maxWidth': '50px'}
            )
        ],style = {"width":"50%","margin":"10px",'marginLeft': "25%", 'marginRight': "25%","textAlign":"center","height":"200px"}
        )

        option_list = [{"label":ele, "value":ele} for ele in values]
        return children_dashtable, option_list, style_DivStep3_report

@app.callback(
    Output(component_id="Div-Case-Study", component_property="style"),
    [Input(component_id = "Button-model-determine", component_property="n_clicks")],
    [State("Dropdown-model-select", "value")]
)
def update_final_model(n_clicks, model_name):
    if n_clicks is None:
        raise PreventUpdate
    else:
        style_DivStep4.update({"display":"inline-block"})
        if model_name=="Logistic Model":
            model_tmp = logitreg_fit(data_upload_dict["data"], xvar_list)[2]
        elif model_name == "Poisson Model":
            model_tmp = poisreg_fit(data_upload_dict["data"], xvar_list)[2]
        
        final_model.update({"name":model_name,"model_fit":model_tmp})
        # print(final_model["name"])
        return style_DivStep4

@app.callback(
    [Output("Graph-survival-curves","figure"),
     Output("Div-casestudy-results","style")],
    [Input("Button-calculator","n_clicks")],
    [State("Input-age","value"),
     State("Input-gender","value"),
     State("Input-smoke","value"),
     State("Input-height","value"),
     State("Input-weight","value"),
     State("Input-sbp","value"),
     State("Input-dbp","value"),
     State("Input-diab","value"),
     State("Input-glyco","value"),
     State("Input-tc","value"),
     State("Input-hdl","value"),
     State("Input-dishis","value")]
)
def update_survivalcurve(n_clicks,age,gender,smoker,h,w,sbp,dbp,diab,gly,tc,hdl,diss):
    if any(ele is None for ele in [n_clicks, age,gender,smoker,h,w,sbp,dbp,diab,gly,tc,hdl]):
        raise PreventUpdate
    else:
        style_DivStep4_survcurve.update({"display":"block"})
        data_tmp = data_upload_dict["data"]
        maxfuy_tmp = data_upload_dict["max_FUY"]
        mdl_tmp = final_model["model_fit"]
        df_tmp = mapping_casestudy(data_tmp,xvar_list,age,gender,smoker,h,w,sbp,dbp,diab,gly,tc,hdl,diss)
        figure = casestudy_results(df_tmp,mdl_tmp,maxfuy_tmp)
        return figure, style_DivStep4_survcurve
    

if __name__ == '__main__':
    app.run_server(debug=True)


