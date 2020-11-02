"""
This model analysis for this py file includes:
    1. seed analysis
    2. prediciton model comparison
"""

import warnings, os
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from tpot import TPOTClassifier
from sklearn.preprocessing import PolynomialFeatures

# Machine learners
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

from pathlib import Path
save_fileloc = os.path.join(Path(__file__).parent, '../dataset/result/')
print(save_fileloc)

class DeepLearning:
    def __init__(self, input_dim, hidden_dim, output_dim):
        model = tf.keras.Sequential()

        # input layer
        model.add(tf.keras.layers.Dense(input_dim, input_shape = (input_dim,), activation = 'relu'))
        # hidden layer
        for dim in hidden_dim:
            model.add(tf.keras.layers.Dense(dim, activation = 'relu'))
        # output layer
        if output_dim == 1:
            model.add(tf.keras.layers.Dense(output_dim, activation = 'sigmoid'))
            model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
        else:
            model.add(tf.keras.layers.Dense(output_dim, activation = 'softmax'))
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 5)]
        self.model = model
    def fit(self, trX, trY, teX, teY):
        self.model.fit(trX, trY, epochs = 100, validation_data = (teX, teY), callbacks = self.callbacks, verbose = 0)
    def predict_proba(self, X):
        return self.model.predict(X)
    def predict(self, X, thres = 0.5):
        return (self.model.predict(X)[:,1] >= thres) * 1

parameters = {
    'LogisticRegression':{'n_jobs': -1},
    'RandomForest':{'n_estimators': 1000, 'n_jobs': -1},
    'SVM': {},
    'best':{'bootstrap':False, 'criterion':"entropy", 'max_features':0.9500000000000001, 'min_samples_leaf':2, 'min_samples_split':2, 'n_estimators':100, 'n_jobs': -1},
    'DeepLearning':{'hidden_dim': [1000,500,100]},
    'GradientBoosting': {}
}

models = {'LogisticRegression': LogisticRegression,
         'RandomForest':RandomForestClassifier,
          'SVM':SVC,
         'best':ExtraTreesClassifier,
         'DeepLearning': DeepLearning,
         'GradientBoosting': GradientBoostingClassifier}

base_dir = '../dataset'

train_df = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))

feature_sets_to_compare = [['target_antibiotics_d7', 'Antianerobes_d7', 'target_antibiotics_d30', 'BL3112', 'Prev_Candidemia', 'Antianerobes_d30', 'Central', 'BL3119', 'prev_other_candida_source', 'lymphoid_haematopoietic', 'Carbapenem_d7', 'BL2021', 'Extended_spectrum_penicillin_d30', 'BL3114', 'mean7_HR', 'Carbapenem_d30', 'TPN_d30', 'HR', 'BL201806', 'mean1_HR', 'var7_RR', 'BL201809', 'var7_SBP', 'steroid_d30', 'Gender', 'BL3118', 'Wt', 'age_at_Cx', 'BL3140', 'CCI'], ['target_antibiotics_d7', 'digestive', 'Antianerobes_d7', 'Tx_surgery', 'Tx_chemo', 'target_antibiotics_d30', 'immunosuppressant_d30', 'BL3112', 'Prev_Candidemia', 'Antianerobes_d30', 'Central', 'BL3119', 'prev_other_candida_source', 'lymphoid_haematopoietic', 'smoking', 'SBP', 'DBP', 'Carbapenem_d7', 'RR', 'BT', 'mean1_SBP', 'mean1_DBP', 'BL2021', 'mean1_RR', 'mean1_BT', 'mean7_SBP', 'mean7_DBP', 'Extended_spectrum_penicillin_d30', 'mean7_RR', 'mean7_BT', 'BL3114', 'var7_DBP', 'var7_HR', 'mean7_HR', 'var7_BT', 'Carbapenem_d30', 'TPN_d30', 'BL201810', 'HR', 'BL201806', 'BL3113', 'mean1_HR', 'var7_RR', 'BL201809', 'BL312002', 'BL3122', 'var7_SBP', 'BL3157', 'steroid_d30', 'Broad_specturm_cephalosporine_3rd_d7', 'Gender', 'Extended_spectrum_penicillin_d7', 'Glycopeptide_d7', 'BL3118', 'Wt', 'Broad_specturm_cephalosporine_3rd_d30', 'age_at_Cx', 'BL3140', 'Glycopeptide_d30', 'CCI', 'Extend_NM_Distant', 'Occupation_NM_기타 NEC, NOS (직업 : 무 & 55세 이상 남자, 여자)'], ['LiverMild', 'LiverSevere', 'DM', 'DMcx', 'target_antibiotics_d7', 'Extend_CD', 'digestive', 'Antianerobes_d7', 'Tx_surgery', 'Tx_chemo', 'Tx_RT', 'target_antibiotics_d30', 'immunosuppressant_d30', 'BL3112', 'OP30', 'Prev_Candidemia', 'Antianerobes_d30', 'Central', 'BL3119', 'prev_other_candida_source', 'lymphoid_haematopoietic', 'Ht', 'smoking', 'drinking', 'SBP', 'DBP', 'Carbapenem_d7', 'RR', 'BT', 'mean1_SBP', 'mean1_DBP', 'BL2021', 'mean1_RR', 'mean1_BT', 'var1_SBP', 'var1_DBP', 'var1_HR', 'var1_RR', 'var1_BT', 'mean7_SBP', 'mean7_DBP', 'Extended_spectrum_penicillin_d30', 'mean7_RR', 'mean7_BT', 'BL3114', 'var7_DBP', 'var7_HR', 'mean7_HR', 'var7_BT', 'BL2011', 'BL2013', 'BL201401', 'BL201402', 'BL201403', 'BL2016', 'Carbapenem_d30', 'TPN_d30', 'BL201810', 'BL201815', 'BL201816', 'BL201818', 'HR', 'BL211103', 'BL3111', 'BL201806', 'BL311201', 'BL3113', 'mean1_HR', 'BL3117', 'var7_RR', 'BL201809', 'BL3120', 'BL312002', 'BL3121', 'BL3122', 'BL3123', 'var7_SBP', 'BL3157', 'BL5044', 'steroid_d30', 'Broad_specturm_cephalosporine_3rd_d7', 'Gender', 'Extended_spectrum_penicillin_d7', 'Glycopeptide_d7', 'BL3118', 'Wt', 'Broad_specturm_cephalosporine_3rd_d30', 'age_at_Cx', 'BL3140', 'Glycopeptide_d30', 'CCI', 'LOS_at_Cx', 'Extend_NM_Distant', 'Extend_NM_Localized', 'Occupation_NM_기타 NEC, NOS (직업 : 무 & 55세 이상 남자, 여자)'], ['MI', 'CHF', 'PVD', 'Stroke', 'Dementia', 'Pulmonary', 'Rheumatic', 'PUD', 'LiverMild', 'LiverSevere', 'DM', 'DMcx', 'Paralysis', 'Renal', 'Cancer', 'Mets', 'HIV', 'age_score', 'target_antibiotics_d7', 'Multiple_Primary', 'Extend_CD', 'Ca_LN_No', 'bone', 'brain_CNS_eye', 'breast', 'digestive', 'female_genital', 'lip_oral_pharynx', 'Antianerobes_d7', 'male_genital', 'respiratory', 'skin', 'soft_tissue', 'thyroid_endocrinegland', 'urinary_tract', 'unspecified', 'Tx_surgery', 'Tx_chemo', 'Tx_RT', 'Tx_hormon', 'Tx_bio', 'Tx_miscellaneous', 'target_antibiotics_d30', 'immunosuppressant_d30', 'BL3112', 'OP30', 'Prev_Candidemia', 'Antianerobes_d30', 'Central', 'BL3119', 'prev_other_candida_source', 'lymphoid_haematopoietic', 'Ht', 'smoking', 'drinking', 'SBP', 'DBP', 'Carbapenem_d7', 'RR', 'BT', 'SpO2', 'mean1_SBP', 'mean1_DBP', 'BL2021', 'mean1_RR', 'mean1_BT', 'mean1_SpO2', 'var1_SBP', 'var1_DBP', 'var1_HR', 'var1_RR', 'var1_BT', 'var1_SpO2', 'mean7_SBP', 'mean7_DBP', 'Extended_spectrum_penicillin_d30', 'mean7_RR', 'mean7_BT', 'mean7_SpO2', 'BL3114', 'var7_DBP', 'var7_HR', 'mean7_HR', 'var7_BT', 'var7_SpO2', 'BL2011', 'BL2012', 'BL2013', 'BL2014', 'BL201401', 'BL201402', 'BL201403', 'BL2015', 'BL2016', 'BL2017', 'BL201701', 'BL201801', 'BL201802', 'BL201803', 'BL201804', 'BL201805', 'Carbapenem_d30', 'BL201807', 'BL201808', 'TPN_d30', 'BL201810', 'BL201811', 'BL201812', 'BL201813', 'BL201814', 'BL201815', 'BL201816', 'BL201817', 'BL201818', 'BL2019', 'BL2020', 'HR', 'BL211103', 'BL3111', 'BL201806', 'BL311201', 'BL3113', 'mean1_HR', 'BL3115', 'BL3116', 'BL3117', 'var7_RR', 'BL201809', 'BL3120', 'BL312002', 'BL3121', 'BL3122', 'BL3123', 'var7_SBP', 'BL3157', 'BL5044', 'steroid_d30', 'Broad_specturm_cephalosporine_3rd_d7', 'Gender', 'Extended_spectrum_penicillin_d7', 'Glycopeptide_d7', 'BL3118', 'Wt', 'Broad_specturm_cephalosporine_3rd_d30', 'age_at_Cx', 'BL3140', 'Glycopeptide_d30', 'CCI', 'LOS_at_Cx', 'Extend_NM_0', 'Extend_NM_Distant', 'Extend_NM_In situ', 'Extend_NM_Localized', 'Extend_NM_Regional direct extension and nodes', 'Extend_NM_Regional nodes only', 'Extend_NM_Regional, NOS', 'Extend_NM_Regional, direct extention only', 'Extend_NM_Unknown if extension or metastasis(unstaged, unknown, or unspecified)', 'Occupation_NM_가정주부 (나이상관없이), 기혼 (55세미만 여자)', 'Occupation_NM_각종 중개인 (부동산, 임대업, 무역)', 'Occupation_NM_경영 관리직 (회사, 금융업 경영자 및 임원, 문화재단이사장)', 'Occupation_NM_경찰관, 소방관', 'Occupation_NM_고급 기술직 (건축기술자, 항공, 선박기술자, 통역사, 정보관리자, 감정평가사)', 'Occupation_NM_공연 예술가', 'Occupation_NM_교육 종사자, 연구원, 학자 (교수, 박사, 교사, 대학시간강사, 강사, 유치원교사)', 'Occupation_NM_국회의원 및 정치인', 'Occupation_NM_기타 (각종 입시준비생, 연수생, 휴학생 등) NEC, NOS', 'Occupation_NM_기타 NEC, NOS', 'Occupation_NM_기타 NEC, NOS (직업 : 무 & 55세 이상 남자, 여자)', 'Occupation_NM_기타 사회 사업가 등 전문적인 활동인 NEC, NOS (세무사, 프리랜서, 디자이너)', 'Occupation_NM_기타 운수 관련 종사자 NEC, NOS', 'Occupation_NM_기타 일반 사무직 근로자 NEC, NOS (은행원, 교직원)', 'Occupation_NM_기타 일반 생산직 근로자 NEC, NOS', 'Occupation_NM_기타 일반 서비스직 종사자 NEC, NOS', 'Occupation_NM_기타 판매 종사자 NEC, NOS', 'Occupation_NM_농업 (농장, 양봉 및 양잠종사자)', 'Occupation_NM_대학, 대학원생', 'Occupation_NM_도, 소매 자영업자 (상업, 사업, 학원운영)', 'Occupation_NM_도, 소매 판매 종사자', 'Occupation_NM_법조계 (판사, 검사, 법무사, 회계사)', 'Occupation_NM_비공연 예술가 (작가, 화가, 영화연출자)', 'Occupation_NM_서비스직 종사자 (조리사, 웨이터, 세탁공, 이발사, 미용사, 안내원, 장의사,수위, 경비원, 여행가이드 등)', 'Occupation_NM_수렵업', 'Occupation_NM_수산업 (해녀)', 'Occupation_NM_실내 제조업 종사자 (재봉공, 제지공, 인쇄공, 가구공, 방송 및 음향 조작공,귀금속 세공공 등)', 'Occupation_NM_요식 숙박업 경영자', 'Occupation_NM_의료계 (의사, 한의사, 치과의사, 수의사, 간호사, 의료기사, 약사, 조산사)', 'Occupation_NM_일반 운송차량 운전사(택시)', 'Occupation_NM_일반 행정 공무원 (우체국, 교육 공무원, 군무원)', 'Occupation_NM_일반사병 (군복무)', 'Occupation_NM_일반장교 (중령, 대령, 중위)', 'Occupation_NM_정부 관리직 공무원 (우체국장, 조합장)', 'Occupation_NM_종교계', 'Occupation_NM_중기계 운전사', 'Occupation_NM_직종을 보고하지 않은 종사자 (직업 : 무 &55세 미만 남자)', 'Occupation_NM_청소부, 가정부 등 노동자에 준하는 자', 'Occupation_NM_축산업', 'Occupation_NM_판매 외무원 (보험설계사)', 'Occupation_NM_하사관', 'Occupation_NM_현장 육체 노동자 (공업, 광원, 목수, 채석원, 정비공, 용접공, 기계 노동자,전기공, 미장 및 도배원 등)', 'Occupation_NM_화물차량 운전사', 'Occupation_NM_회사 사무원', 'Marriage_0', 'Marriage_기타', 'Marriage_기혼', 'Marriage_미혼', 'Marriage_별거', 'Marriage_사별', 'Marriage_유', 'Marriage_이혼']]

record_dict = {
    'LogisticRegression': {'model': models['LogisticRegression'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['LogisticRegression']},
    'RandomForest':{'model': models['RandomForest'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['RandomForest']},
    'SVM':{'model': models['SVM'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['SVM']},
    'best':{'model': models['best'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['best']},
    'GradientBoosting': {'model': models['GradientBoosting'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['best']},
    'DeepLearning':{'model': models['DeepLearning'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['DeepLearning']}
}

record_dict = {'best':{'model': models['best'], 'seed': [], 'var': [], 'auc': [], 'tn': [], 'thres': [], 'fp': [], 'fn': [], 'tp': [], 'parameters': parameters['best']}}

list_seed = np.arange(1,101)
for seed in list_seed:
    print(seed)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    for i, list_idx in enumerate(feature_sets_to_compare):
        trX = train_df[list_idx].values
        trY = train_df['case_control'].values
        teX = test_df[list_idx].values
        teY = test_df['case_control'].values
        for model_name in record_dict:
            model_initialize_func = record_dict[model_name]['model']
            model_parameters = record_dict[model_name]['parameters']
            if model_name != 'DeepLearning':
                model_parameters['random_state'] = seed
            if model_name == 'DeepLearning':
                model_parameters['input_dim'] = trX.shape[1]
                model_parameters['output_dim'] = 1 if len(trY.shape) == 1 else trY.shape[1]
            clf = model_initialize_func(**model_parameters)
            if model_name == 'DeepLearning':
                clf.fit(trX, trY, teX, teY)
            else:
                clf.fit(trX, trY)
            if 'thres' in record_dict[model_name].keys():
                if model_name == 'DeepLearning':
                    probas = clf.predict_proba(teX).flatten()
                else:
                    probas = clf.predict_proba(teX)[:,1]
                fpr, tpr, thres = roc_curve(teY, probas)
                auc_score = auc(fpr, tpr)
                for t in thres:
                    pred = (probas >= t) * 1
                    tn, fp, fn, tp = confusion_matrix(y_true = teY, y_pred = pred).ravel()
                    record_dict[model_name]['seed'].append(seed)
                    record_dict[model_name]['var'].append(i+1)
                    record_dict[model_name]['auc'].append(auc_score)
                    record_dict[model_name]['thres'].append(t)
                    record_dict[model_name]['tn'].append(tn); record_dict[model_name]['fp'].append(fp); record_dict[model_name]['fn'].append(fn); record_dict[model_name]['tp'].append(tp)
            else:
                pred = clf.predict(teX)
                fpr, tpr, thres = roc_curve(teY, pred)
                auc_score = auc(fpr, tpr)
                tn, fp, fn, tp = confusion_matrix(y_true = teY, y_pred = pred).ravel()
                record_dict[model_name]['seed'].append(seed)
                record_dict[model_name]['var'].append(i+1)
                record_dict[model_name]['auc'].append(auc_score)
                record_dict[model_name]['tn'].append(tn); record_dict[model_name]['fp'].append(fp); record_dict[model_name]['fn'].append(fn); record_dict[model_name]['tp'].append(tp)
                
report_df_dict = {}
record_cols = ['seed', 'var', 'auc', 'thres', 'tp', 'fp', 'tn', 'fn']
for model_name in record_dict:
    report_df_dict[model_name] = pd.DataFrame()
    for col in record_cols:
        if col in record_dict[model_name].keys():
            report_df_dict[model_name][col] = record_dict[model_name][col]
            
for model_name in report_df_dict:
    report_df_dict[model_name]['sensitivity'] = report_df_dict[model_name]['tp'] / (report_df_dict[model_name]['tp'] + report_df_dict[model_name]['fn'])
    report_df_dict[model_name]['specificity'] = report_df_dict[model_name]['tn'] / (report_df_dict[model_name]['tn'] + report_df_dict[model_name]['fp'])
    report_df_dict[model_name]['ppv (precision)'] = report_df_dict[model_name]['tp'] / (report_df_dict[model_name]['tp'] + report_df_dict[model_name]['fp'])
    report_df_dict[model_name]['npv'] = report_df_dict[model_name]['tn'] / (report_df_dict[model_name]['tn'] + report_df_dict[model_name]['fn'])
    report_df_dict[model_name]['f1'] = 2 * (report_df_dict[model_name]['ppv (precision)'] * report_df_dict[model_name]['sensitivity']) / (report_df_dict[model_name]['ppv (precision)'] + report_df_dict[model_name]['sensitivity'])


report_df_dict['LogisticRegression'].to_excel(os.path.join(save_fileloc, 'logistic.xlsx'))
report_df_dict['RandomForest'].to_excel(os.path.join(save_fileloc, 'randomforest.xlsx'))
report_df_dict['GradientBoosting'].to_excel(os.path.join(save_fileloc, 'gradientboosting.xlsx'))
report_df_dict['SVM'].to_excel(os.path.join(save_fileloc, 'svm.xlsx'))
report_df_dict['best'].to_excel(os.path.join(save_fileloc, 'best.xlsx'))
report_df_dict['DeepLearning'].to_excel(os.path.join(save_fileloc, 'nn.xlsx'))
